"""
HTTP request handler for the annotation interface.
"""

import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from .state import (
    STATE, get_session, get_session_token_from_cookie, create_session,
    check_failed_attempt_lockout, record_failed_attempt, clear_failed_attempts,
    hash_password
)
from .data import make_turn_key, load_file_for_annotation, save_all_extractions
from .pages import get_login_page, get_file_picker_page, get_annotation_page


class AnnotationHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the annotation interface."""
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def get_client_ip(self):
        """Get client IP address."""
        # Check for X-Forwarded-For header (behind proxy)
        forwarded = self.headers.get('X-Forwarded-For')
        if forwarded:
            return forwarded.split(',')[0].strip()
        return self.client_address[0]
    
    def get_session(self):
        """Get the current user's session, or None if not authenticated."""
        cookie = self.headers.get('Cookie')
        token = get_session_token_from_cookie(cookie)
        return get_session(token)
    
    def is_authenticated(self):
        """Check if the current request is authenticated."""
        # If no password is set, create/get anonymous session
        if STATE["password_hash"] is None:
            cookie = self.headers.get('Cookie')
            token = get_session_token_from_cookie(cookie)
            if not token or not get_session(token):
                # Will need to create session on first request
                return True  # Allow through, session created in handler
            return True
        # Check session cookie
        return self.get_session() is not None
    
    def ensure_session(self):
        """Ensure a session exists, creating one if needed (for no-password mode)."""
        session = self.get_session()
        if session is None and STATE["password_hash"] is None:
            # No password mode - create anonymous session
            token = create_session()
            session = get_session(token)
            # Return the token so we can set the cookie
            return session, token
        return session, None
    
    def send_login_redirect(self):
        """Redirect to login page."""
        self.send_response(302)
        self.send_header('Location', '/login')
        self.end_headers()
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        # Login page - always accessible
        if parsed.path == '/login':
            # If already authenticated, redirect to main page
            if self.is_authenticated():
                self.send_response(302)
                self.send_header('Location', '/')
                self.end_headers()
                return
            
            client_ip = self.get_client_ip()
            wait_time = check_failed_attempt_lockout(client_ip)
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(get_login_page(wait_time=wait_time).encode('utf-8'))
            return
        
        # Logout
        if parsed.path == '/logout':
            # Clear session cookie
            self.send_response(302)
            self.send_header('Location', '/login')
            self.send_header('Set-Cookie', 'session=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict')
            self.end_headers()
            return
        
        # All other routes require authentication
        if not self.is_authenticated():
            self.send_login_redirect()
            return
        
        # Get or create session
        session, new_token = self.ensure_session()
        
        # In server mode without a loaded file, show file picker
        if parsed.path == '/' or parsed.path == '':
            if STATE["server_mode"] and not session.file_loaded:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                if new_token:
                    self.send_header('Set-Cookie', f'session={new_token}; Path=/; HttpOnly; SameSite=Strict')
                self.end_headers()
                self.wfile.write(get_file_picker_page().encode('utf-8'))
            else:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                if new_token:
                    self.send_header('Set-Cookie', f'session={new_token}; Path=/; HttpOnly; SameSite=Strict')
                self.end_headers()
                self.wfile.write(get_annotation_page(session).encode('utf-8'))
        
        elif parsed.path == '/annotate':
            # Direct annotation page access
            if not session.file_loaded:
                self.send_response(302)
                self.send_header('Location', '/')
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(get_annotation_page(session).encode('utf-8'))
        
        elif parsed.path == '/change_file':
            # Go back to file picker (server mode only) - reset session's file state
            if STATE["server_mode"]:
                session.file_loaded = False
                session.conversations = []
                session.assistant_turns = []
                session.extracted_claims = {}
                session.current_flat_index = 0
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
            
        elif parsed.path == '/navigate':
            params = parse_qs(parsed.query)
            action = params.get('action', [''])[0]
            
            if action == 'first':
                session.current_flat_index = 0
            elif action == 'last':
                session.current_flat_index = len(session.assistant_turns) - 1
            elif action == 'prev':
                session.current_flat_index = max(0, session.current_flat_index - 1)
            elif action == 'next':
                session.current_flat_index = min(len(session.assistant_turns) - 1, session.current_flat_index + 1)
            elif action == 'jump':
                index = int(params.get('index', [0])[0])
                session.current_flat_index = max(0, min(len(session.assistant_turns) - 1, index))
            elif action == 'next_empty':
                # Find next turn without claims
                for i in range(session.current_flat_index + 1, len(session.assistant_turns)):
                    turn = session.assistant_turns[i]
                    key = make_turn_key(turn['conv_idx'], turn['turn_number'])
                    if key not in session.extracted_claims or len(session.extracted_claims[key]) == 0:
                        session.current_flat_index = i
                        break
                else:
                    # Wrap around
                    for i in range(0, session.current_flat_index):
                        turn = session.assistant_turns[i]
                        key = make_turn_key(turn['conv_idx'], turn['turn_number'])
                        if key not in session.extracted_claims or len(session.extracted_claims[key]) == 0:
                            session.current_flat_index = i
                            break
            
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
            
        elif parsed.path == '/get_claim':
            params = parse_qs(parsed.query)
            index = int(params.get('index', [0])[0])
            
            turn = session.assistant_turns[session.current_flat_index]
            key = make_turn_key(turn['conv_idx'], turn['turn_number'])
            claims = session.extracted_claims.get(key, [])
            
            claim = claims[index] if 0 <= index < len(claims) else {}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(claim).encode('utf-8'))
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        # Handle login POST
        if self.path == '/login':
            client_ip = self.get_client_ip()
            
            # Check for lockout - if locked out, just show wait message (don't process)
            wait_time = check_failed_attempt_lockout(client_ip)
            if wait_time > 0:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(get_login_page(
                    error_msg="Too many failed attempts. Please wait.",
                    wait_time=wait_time
                ).encode('utf-8'))
                return
            
            # Parse form data
            params = parse_qs(post_data)
            password = params.get('password', [''])[0]
            
            # Verify password
            if hash_password(password) == STATE["password_hash"]:
                # Success - create session
                token = create_session()
                clear_failed_attempts(client_ip)
                
                self.send_response(302)
                self.send_header('Location', '/')
                self.send_header('Set-Cookie', f'session={token}; Path=/; HttpOnly; SameSite=Strict')
                self.end_headers()
            else:
                # Failed - record attempt and show error with wait time
                record_failed_attempt(client_ip)
                new_wait = check_failed_attempt_lockout(client_ip)
                
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(get_login_page(
                    error_msg="Incorrect password. Please try again.",
                    wait_time=new_wait
                ).encode('utf-8'))
            return
        
        # All other POST routes require authentication
        if not self.is_authenticated():
            self.send_response(401)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Unauthorized"}).encode('utf-8'))
            return
        
        # Get session for authenticated routes
        session, _ = self.ensure_session()
        
        if self.path == '/load_files':
            # Handle user identification and file loading
            try:
                data = json.loads(post_data)
                username = data.get('username', '').strip()
                
                if not username:
                    response = {"success": False, "error": "Please enter your name."}
                elif not username.replace('_', '').replace('-', '').isalnum():
                    response = {"success": False, "error": "Name can only contain letters, numbers, underscores, and hyphens."}
                else:
                    # Fixed input file: responses.jsonl
                    input_filename = "responses.jsonl"
                    # Output file based on username
                    output_filename = f"responses_{username}_annotations.jsonl"
                    
                    # Construct full paths
                    input_path = os.path.join(STATE["data_dir"], input_filename)
                    output_path = os.path.join(STATE["data_dir"], output_filename)
                    
                    if not os.path.exists(input_path):
                        response = {"success": False, "error": f"Input file not found: {input_filename}. Please ensure responses.jsonl exists in the data directory."}
                    else:
                        success, error = load_file_for_annotation(session, input_path, output_path)
                        if success:
                            session.username = username  # Store username in session
                            response = {"success": True}
                        else:
                            response = {"success": False, "error": error}
            except json.JSONDecodeError:
                response = {"success": False, "error": "Invalid request format."}
            except Exception as e:
                response = {"success": False, "error": str(e)}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        
        elif self.path == '/save_claim':
            data = json.loads(post_data)
            
            turn = session.assistant_turns[session.current_flat_index]
            key = make_turn_key(turn['conv_idx'], turn['turn_number'])
            
            if key not in session.extracted_claims:
                session.extracted_claims[key] = []
            
            claim = {
                "inferred_source_type": data.get('inferred_source_type', 'paper'),
                "claimed_content": data.get('claimed_content', ''),
                "claimed_title": data.get('claimed_title', ''),
                "claimed_authors": data.get('claimed_authors', ''),
                "claimed_year": data.get('claimed_year', ''),
                "claimed_institution": data.get('claimed_institution', ''),
                "claimed_url": data.get('claimed_url', ''),
                "human_reference_grounding": data.get('human_reference_grounding', ''),
                "human_content_grounding": data.get('human_content_grounding', ''),
                "human_comment": data.get('human_comment', '')
            }
            
            editing_index = data.get('editing_index', -1)
            if editing_index >= 0 and editing_index < len(session.extracted_claims[key]):
                session.extracted_claims[key][editing_index] = claim
            else:
                session.extracted_claims[key].append(claim)
            
            # Save to file
            save_all_extractions(session)
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
            
        elif self.path == '/delete_claim':
            params = parse_qs(post_data)
            index = int(params.get('index', [0])[0])
            
            turn = session.assistant_turns[session.current_flat_index]
            key = make_turn_key(turn['conv_idx'], turn['turn_number'])
            
            if key in session.extracted_claims and 0 <= index < len(session.extracted_claims[key]):
                session.extracted_claims[key].pop(index)
                save_all_extractions(session)
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
            
        else:
            self.send_response(404)
            self.end_headers()

