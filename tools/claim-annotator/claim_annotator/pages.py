"""
HTML page generators for the annotation interface.
"""

import html as html_module
import os
from datetime import datetime

from .state import STATE
from .data import make_turn_key


def get_login_page(error_msg="", wait_time=0):
    """Generate the login page HTML."""
    error_html = f'<div class="error-msg">{html_module.escape(error_msg)}</div>' if error_msg else ''
    wait_script = f'''
        <script>
            let waitTime = {int(wait_time)};
            const btn = document.getElementById('loginBtn');
            const countdownEl = document.getElementById('countdown');
            
            function updateCountdown() {{
                if (waitTime > 0) {{
                    btn.disabled = true;
                    countdownEl.style.display = 'block';
                    countdownEl.textContent = 'Please wait ' + waitTime + ' seconds...';
                    waitTime--;
                    setTimeout(updateCountdown, 1000);
                }} else {{
                    btn.disabled = false;
                    countdownEl.style.display = 'none';
                }}
            }}
            updateCountdown();
        </script>
    ''' if wait_time > 0 else ''
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Claim Annotation Tool</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --bg-elevated: #21262d;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-cyan: #39c5cf;
            --accent-purple: #a371f7;
            --accent-red: #f85149;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Space Grotesk', -apple-system, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .login-container {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 40px;
            width: 100%;
            max-width: 400px;
            text-align: center;
        }}
        
        .login-title {{
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }}
        
        .login-subtitle {{
            color: var(--text-secondary);
            margin-bottom: 32px;
        }}
        
        .form-group {{
            margin-bottom: 20px;
            text-align: left;
        }}
        
        .form-label {{
            display: block;
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 8px;
            color: var(--text-primary);
        }}
        
        .form-input {{
            width: 100%;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 12px 16px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 1rem;
        }}
        
        .form-input:focus {{
            outline: none;
            border-color: var(--accent-cyan);
        }}
        
        .login-btn {{
            width: 100%;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            border: none;
            color: white;
            padding: 14px 24px;
            border-radius: 10px;
            cursor: pointer;
            font-family: inherit;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.2s ease;
            margin-top: 8px;
        }}
        
        .login-btn:hover:not(:disabled) {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(57, 197, 207, 0.3);
        }}
        
        .login-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}
        
        .error-msg {{
            background: rgba(248, 81, 73, 0.15);
            border: 1px solid var(--accent-red);
            color: var(--accent-red);
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }}
        
        .countdown {{
            color: var(--accent-red);
            font-size: 0.875rem;
            margin-top: 12px;
            display: none;
        }}
        
        .lock-icon {{
            font-size: 3rem;
            margin-bottom: 16px;
        }}
    </style>
</head>
<body>
    <div class="login-container">
        <div class="lock-icon">🔐</div>
        <h1 class="login-title">Claim Annotation</h1>
        <p class="login-subtitle">Enter password to continue</p>
        
        {error_html}
        
        <form method="POST" action="/login">
            <div class="form-group">
                <label class="form-label" for="password">Password</label>
                <input type="password" class="form-input" id="password" name="password" 
                       placeholder="Enter password" required autofocus>
            </div>
            <button type="submit" class="login-btn" id="loginBtn">Login →</button>
            <div class="countdown" id="countdown"></div>
        </form>
    </div>
    {wait_script}
</body>
</html>'''


def get_file_picker_page():
    """Generate the user identification page for annotation."""
    data_dir = STATE["data_dir"]
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome - Claim Annotation Tool</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --bg-elevated: #21262d;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-purple: #a371f7;
            --accent-orange: #db6d28;
            --accent-cyan: #39c5cf;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Space Grotesk', -apple-system, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 500px;
            width: 100%;
            padding: 24px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        
        .header h1 {{
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }}
        
        .header p {{
            color: var(--text-secondary);
            font-size: 1rem;
        }}
        
        .welcome-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px;
        }}
        
        .form-group {{
            margin-bottom: 24px;
        }}
        
        .form-label {{
            display: block;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-primary);
        }}
        
        .form-label .icon {{
            margin-right: 8px;
        }}
        
        .form-input {{
            width: 100%;
            background: var(--bg-elevated);
            border: 2px solid var(--border);
            color: var(--text-primary);
            padding: 14px 16px;
            border-radius: 10px;
            font-family: inherit;
            font-size: 1rem;
            transition: border-color 0.2s ease;
        }}
        
        .form-input:focus {{
            outline: none;
            border-color: var(--accent-cyan);
        }}
        
        .form-input::placeholder {{
            color: var(--text-secondary);
        }}
        
        .hint {{
            margin-top: 12px;
            padding: 12px 16px;
            background: rgba(210, 153, 34, 0.1);
            border: 1px solid rgba(210, 153, 34, 0.3);
            border-radius: 8px;
            font-size: 0.875rem;
            color: var(--accent-yellow);
        }}
        
        .hint .hint-icon {{
            margin-right: 6px;
        }}
        
        .start-btn {{
            width: 100%;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            border: none;
            color: white;
            padding: 16px 40px;
            border-radius: 10px;
            cursor: pointer;
            font-family: inherit;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.2s ease;
            margin-top: 8px;
        }}
        
        .start-btn:hover:not(:disabled) {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(57, 197, 207, 0.3);
        }}
        
        .start-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }}
        
        .error-msg {{
            background: rgba(248, 81, 73, 0.15);
            border: 1px solid var(--accent-red);
            color: var(--accent-red);
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
            font-size: 0.9rem;
        }}
        
        .error-msg.show {{
            display: block;
        }}
        
        .data-dir-info {{
            text-align: center;
            margin-top: 20px;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        
        .data-dir-info code {{
            background: var(--bg-elevated);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Claim Annotation Tool</h1>
            <p>Enter your name to start or resume annotation</p>
        </div>
        
        <div class="welcome-card">
            <div class="error-msg" id="errorMsg"></div>
            
            <div class="form-group">
                <label class="form-label" for="userName">
                    <span class="icon">👤</span>Your Name
                </label>
                <input type="text" class="form-input" id="userName" 
                       placeholder="Enter your name (e.g., John)" 
                       autocomplete="off" autofocus>
                
                <div class="hint">
                    <span class="hint-icon">⚠️</span>
                    <strong>Important:</strong> Use the same name each time to resume your previous work. 
                    Your annotations are saved under this name.
                </div>
            </div>
            
            <button class="start-btn" id="startBtn" onclick="startAnnotation()" disabled>
                Start Annotation →
            </button>
        </div>
        
        <div class="data-dir-info">
            Data directory: <code>{html_module.escape(data_dir)}</code>
        </div>
    </div>
    
    <script>
        const userNameInput = document.getElementById('userName');
        const startBtn = document.getElementById('startBtn');
        
        function updateStartButton() {{
            const name = userNameInput.value.trim();
            startBtn.disabled = !name;
        }}
        
        userNameInput.addEventListener('input', updateStartButton);
        userNameInput.addEventListener('keypress', function(e) {{
            if (e.key === 'Enter' && !startBtn.disabled) {{
                startAnnotation();
            }}
        }});
        
        function startAnnotation() {{
            const userName = userNameInput.value.trim();
            if (!userName) {{
                showError('Please enter your name.');
                return;
            }}
            
            // Validate name (alphanumeric, underscores, hyphens only)
            if (!/^[a-zA-Z0-9_-]+$/.test(userName)) {{
                showError('Name can only contain letters, numbers, underscores, and hyphens.');
                return;
            }}
            
            // Disable button while loading
            startBtn.disabled = true;
            startBtn.textContent = 'Loading...';
            
            // POST to load files with username
            fetch('/load_files', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    username: userName
                }})
            }})
            .then(r => r.json())
            .then(data => {{
                if (data.success) {{
                    window.location.href = '/annotate';
                }} else {{
                    showError(data.error || 'Failed to load files.');
                    startBtn.disabled = false;
                    startBtn.textContent = 'Start Annotation →';
                }}
            }})
            .catch(err => {{
                showError('Network error: ' + err.message);
                startBtn.disabled = false;
                startBtn.textContent = 'Start Annotation →';
            }});
        }}
        
        function showError(msg) {{
            const el = document.getElementById('errorMsg');
            el.textContent = msg;
            el.classList.add('show');
            setTimeout(() => el.classList.remove('show'), 5000);
        }}
    </script>
</body>
</html>'''


def get_annotation_page(session):
    """Generate the main annotation page HTML for a specific user session."""
    assistant_turns = session.assistant_turns
    current = session.current_flat_index
    extractions = session.extracted_claims
    
    if not assistant_turns:
        return "<html><body><h1>No assistant turns found in the conversations.</h1></body></html>"
    
    # Calculate progress - count turns that have at least one claim
    def has_claims(turn):
        key = make_turn_key(turn['conv_idx'], turn['turn_number'])
        return key in extractions and len(extractions[key]) > 0
    annotated_count = sum(1 for turn in assistant_turns if has_claims(turn))
    total_count = len(assistant_turns)
    progress_pct = (annotated_count / total_count * 100) if total_count > 0 else 0
    
    # Current turn
    turn = assistant_turns[current]
    key = make_turn_key(turn['conv_idx'], turn['turn_number'])
    current_claims = extractions.get(key, [])
    
    # Escape content
    response_escaped = html_module.escape(turn["content"])
    question_escaped = html_module.escape(turn["user_question"])
    
    # Build claims list HTML
    claims_html = ""
    for i, claim in enumerate(current_claims):
        ref_grounding = claim.get('human_reference_grounding', '')
        content_grounding = claim.get('human_content_grounding', '')
        
        ref_badge = f'<span class="mini-badge ref-{ref_grounding}">{ref_grounding}</span>' if ref_grounding else ''
        content_badge = f'<span class="mini-badge content-{content_grounding}">{content_grounding}</span>' if content_grounding else ''
        
        claims_html += f'''
        <div class="claim-item" data-index="{i}">
            <div class="claim-header">
                <span class="claim-number">Claim #{i + 1}</span>
                <span class="claim-type">{html_module.escape(claim.get('inferred_source_type', ''))}</span>
                {ref_badge}
                {content_badge}
                <button class="edit-btn" onclick="editClaim({i})">✎ Edit</button>
                <button class="delete-btn" onclick="deleteClaim({i})">✕</button>
            </div>
            <div class="claim-content">{html_module.escape(claim.get('claimed_content', ''))}</div>
            <div class="claim-meta">
                {f"<span>📄 {html_module.escape(claim.get('claimed_title', ''))}</span>" if claim.get('claimed_title') else ''}
                {f"<span>👤 {html_module.escape(claim.get('claimed_authors', ''))}</span>" if claim.get('claimed_authors') else ''}
                {f"<span>📅 {html_module.escape(str(claim.get('claimed_year', '')))}</span>" if claim.get('claimed_year') else ''}
            </div>
        </div>
        '''
    
    # Add the "Add Claim" button at the end of the claims list
    claims_html += '''
        <button class="add-claim-btn-inline" onclick="openAddForm()">+ Add New Claim</button>
    '''
    
    if len(current_claims) == 0:
        claims_html = '''<div class="no-claims">No claims extracted yet. Click the button below to start.</div>
        <button class="add-claim-btn-inline" onclick="openAddForm()">+ Add New Claim</button>
    '''
    
    # Server mode: add switch user button and show username
    change_file_btn = ''
    username_display = ''
    if STATE["server_mode"]:
        change_file_btn = '<a href="/change_file" class="change-file-btn">👤 Switch User</a>'
        if hasattr(session, 'username') and session.username:
            username_display = f'<span class="username-badge">👤 {html_module.escape(session.username)}</span>'
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claim Extraction Tool</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --bg-elevated: #21262d;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-purple: #a371f7;
            --accent-orange: #db6d28;
            --accent-cyan: #39c5cf;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Space Grotesk', -apple-system, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }}
        
        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }}
        
        .header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .file-info {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            background: var(--bg-elevated);
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid var(--border);
        }}
        
        /* Progress Bar */
        .progress-section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
        }}
        
        .progress-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .progress-text {{
            font-size: 0.875rem;
            color: var(--text-secondary);
        }}
        
        .progress-text strong {{
            color: var(--accent-cyan);
        }}
        
        .progress-bar {{
            height: 8px;
            background: var(--bg-elevated);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        /* Navigation */
        .nav-row {{
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .nav-btn {{
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.875rem;
            transition: all 0.15s ease;
        }}
        
        .nav-btn:hover:not(:disabled) {{
            background: var(--border);
            border-color: var(--accent-cyan);
        }}
        
        .nav-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}
        
        .nav-input {{
            width: 70px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            text-align: center;
        }}
        
        .nav-input:focus {{
            outline: none;
            border-color: var(--accent-cyan);
        }}
        
        /* Main Layout */
        .main-layout {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            align-items: stretch;
        }}
        
        @media (max-width: 1200px) {{
            .main-layout {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* Response Card */
        .response-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .response-header {{
            background: var(--bg-elevated);
            padding: 14px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .response-id {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            color: var(--accent-purple);
        }}
        
        .response-body {{
            padding: 20px;
        }}
        
        .section-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            margin-bottom: 8px;
            font-weight: 600;
        }}
        
        .question-text {{
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent-yellow);
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 16px;
            min-height: 60px;
            max-height: 300px;
            overflow-y: auto;
            resize: vertical;
        }}
        
        .response-text {{
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent-cyan);
            padding: 16px;
            border-radius: 8px;
            font-size: 0.9rem;
            line-height: 1.7;
            white-space: pre-wrap;
            min-height: 100px;
            height: 250px;
            overflow-y: auto;
            resize: vertical;
        }}
        
        /* Claims Panel */
        .claims-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        
        /* Tab Navigation */
        .tab-nav {{
            display: flex;
            background: var(--bg-elevated);
            border-bottom: 1px solid var(--border);
        }}
        
        .tab-btn {{
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 14px 20px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.15s ease;
            position: relative;
        }}
        
        .tab-btn:hover {{
            color: var(--text-primary);
            background: rgba(57, 197, 207, 0.05);
        }}
        
        .tab-btn.active {{
            color: var(--accent-cyan);
            background: var(--bg-card);
        }}
        
        .tab-btn.active::after {{
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        }}
        
        .tab-btn .tab-count {{
            font-size: 0.7rem;
            background: var(--bg-dark);
            padding: 2px 6px;
            border-radius: 10px;
            margin-left: 6px;
        }}
        
        .tab-btn.active .tab-count {{
            background: rgba(57, 197, 207, 0.2);
            color: var(--accent-cyan);
        }}
        
        /* Tab Content */
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .claims-list {{
            padding: 16px;
        }}
        
        .claim-form {{
            padding: 16px;
        }}
        
        .form-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--accent-purple);
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }}
        
        .form-actions {{
            display: flex;
            gap: 12px;
            margin-top: 20px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }}
        
        .no-claims {{
            text-align: center;
            color: var(--text-secondary);
            padding: 40px 20px;
            font-style: italic;
        }}
        
        .add-claim-btn-inline {{
            width: 100%;
            background: var(--bg-elevated);
            border: 2px dashed var(--border);
            color: var(--text-secondary);
            padding: 14px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s ease;
            margin-top: 8px;
        }}
        
        .add-claim-btn-inline:hover {{
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            background: rgba(57, 197, 207, 0.05);
        }}
        
        .claim-item {{
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 12px;
        }}
        
        .claim-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }}
        
        .claim-number {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--accent-purple);
            font-weight: 600;
        }}
        
        .claim-type {{
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(88, 166, 255, 0.15);
            color: var(--accent-blue);
            text-transform: uppercase;
        }}
        
        .mini-badge {{
            font-size: 0.65rem;
            padding: 2px 6px;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        
        .mini-badge.ref-yes, .mini-badge.content-yes {{
            background: rgba(63, 185, 80, 0.15);
            color: var(--accent-green);
        }}
        
        .mini-badge.ref-no, .mini-badge.content-no {{
            background: rgba(248, 81, 73, 0.15);
            color: var(--accent-red);
        }}
        
        .mini-badge.ref-unsure, .mini-badge.content-unsure {{
            background: rgba(163, 113, 247, 0.15);
            color: var(--accent-purple);
        }}
        
        .edit-btn, .delete-btn {{
            margin-left: auto;
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
            transition: all 0.15s ease;
        }}
        
        .edit-btn:hover {{
            border-color: var(--accent-blue);
            color: var(--accent-blue);
        }}
        
        .delete-btn {{
            margin-left: 4px;
        }}
        
        .delete-btn:hover {{
            border-color: var(--accent-red);
            color: var(--accent-red);
        }}
        
        .claim-content {{
            font-size: 0.85rem;
            color: var(--text-primary);
            margin-bottom: 8px;
            line-height: 1.5;
        }}
        
        .claim-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        
        .form-group {{
            margin-bottom: 16px;
        }}
        
        .form-label {{
            display: block;
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 6px;
            color: var(--text-primary);
        }}
        
        .form-label small {{
            font-weight: 400;
            color: var(--text-secondary);
        }}
        
        .form-input, .form-select, .form-textarea {{
            width: 100%;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 10px 12px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.875rem;
        }}
        
        .form-input:focus, .form-select:focus, .form-textarea:focus {{
            outline: none;
            border-color: var(--accent-cyan);
        }}
        
        .form-textarea {{
            min-height: 100px;
            resize: vertical;
        }}
        
        .form-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}
        
        .form-row-3 {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
        }}
        
        /* Annotation Section */
        .annotation-section {{
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            margin-top: 16px;
        }}
        
        .annotation-title {{
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--accent-purple);
        }}
        
        .options-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }}
        
        .option-btn {{
            background: var(--bg-dark);
            border: 2px solid var(--border);
            color: var(--text-secondary);
            padding: 10px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.15s ease;
            text-align: center;
        }}
        
        .option-btn:hover {{
            border-color: var(--accent-cyan);
            color: var(--text-primary);
        }}
        
        .option-btn.selected {{
            border-color: var(--accent-cyan);
            background: rgba(57, 197, 207, 0.15);
            color: var(--accent-cyan);
        }}
        
        .option-btn.yes.selected {{
            border-color: var(--accent-green);
            background: rgba(63, 185, 80, 0.15);
            color: var(--accent-green);
        }}
        
        .option-btn.no.selected {{
            border-color: var(--accent-red);
            background: rgba(248, 81, 73, 0.15);
            color: var(--accent-red);
        }}
        
        .option-btn.unsure.selected {{
            border-color: var(--accent-purple);
            background: rgba(163, 113, 247, 0.15);
            color: var(--accent-purple);
        }}
        
        .btn-clear {{
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.875rem;
            flex: 1;
        }}
        
        .btn-clear:hover {{
            border-color: var(--accent-red);
            color: var(--accent-red);
        }}
        
        .btn-save {{
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            border: none;
            color: white;
            padding: 10px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.875rem;
            font-weight: 600;
            flex: 2;
        }}
        
        .btn-save:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(57, 197, 207, 0.3);
        }}
        
        /* Toast notification */
        .toast {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: var(--accent-green);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 500;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s ease;
            z-index: 2000;
        }}
        
        .toast.show {{
            opacity: 1;
            transform: translateY(0);
        }}
        
        /* Shortcuts */
        .shortcuts {{
            margin-top: 20px;
            padding: 12px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        
        .shortcuts kbd {{
            background: var(--bg-dark);
            border: 1px solid var(--border);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            margin: 0 2px;
        }}
        
        .change-file-btn {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 6px;
            text-decoration: none;
            transition: all 0.15s ease;
            margin-right: 12px;
        }}
        
        .change-file-btn:hover {{
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }}
        
        .header-right {{
            display: flex;
            align-items: center;
        }}
        
        .username-badge {{
            font-size: 0.85rem;
            color: var(--accent-green);
            background: rgba(63, 185, 80, 0.1);
            border: 1px solid rgba(63, 185, 80, 0.3);
            padding: 6px 12px;
            border-radius: 6px;
            margin-right: 12px;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Claim Extraction Tool</h1>
            <div class="header-right">
                {username_display}
                {change_file_btn}
                <div class="file-info">{html_module.escape(os.path.basename(session.input_file))}</div>
            </div>
        </div>
        
        <div class="progress-section">
            <div class="progress-header">
                <span class="progress-text">
                    <strong>{annotated_count}</strong> of <strong>{total_count}</strong> responses annotated ({progress_pct:.1f}%)
                </span>
                <div class="nav-row">
                    <button class="nav-btn" onclick="navigate('first')" {"disabled" if current == 0 else ""}>⏮ First</button>
                    <button class="nav-btn" onclick="navigate('prev')" {"disabled" if current == 0 else ""}>← Prev</button>
                    <input type="number" class="nav-input" id="jumpInput" value="{current + 1}" min="1" max="{total_count}" onchange="jumpTo(this.value)">
                    <span class="progress-text">/ {total_count}</span>
                    <button class="nav-btn" onclick="navigate('next')" {"disabled" if current >= total_count - 1 else ""}>Next →</button>
                    <button class="nav-btn" onclick="navigate('last')" {"disabled" if current >= total_count - 1 else ""}>Last ⏭</button>
                    <button class="nav-btn" onclick="navigate('next_empty')">Next Empty ⏩</button>
                </div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_pct}%"></div>
            </div>
        </div>
        
        <div class="main-layout">
            <div class="response-card">
                <div class="response-header">
                    <span class="response-id">Conv {turn['conversation_id']} • Turn {turn['turn_number']}</span>
                </div>
                <div class="response-body">
                    <div class="section-label">User Question</div>
                    <div class="question-text">{question_escaped}</div>
                    
                    <div class="section-label">Assistant Response</div>
                    <div class="response-text" id="responseText">{response_escaped}</div>
                </div>
            </div>
            
            <div class="claims-panel">
                <div class="tab-nav">
                    <button class="tab-btn active" onclick="switchTab('claims')" id="tabClaims">
                        📋 Claims <span class="tab-count">{len(current_claims)}</span>
                    </button>
                    <button class="tab-btn" onclick="switchTab('form')" id="tabForm">
                        ✏️ <span id="formTabLabel">Add Claim</span>
                    </button>
                </div>
                
                <!-- Claims List Tab -->
                <div class="tab-content active" id="tabContentClaims">
                    <div class="claims-list" id="claimsList">
                        {claims_html}
                    </div>
                </div>
                
                <!-- Add/Edit Claim Form Tab -->
                <div class="tab-content" id="tabContentForm">
                    <div class="claim-form">
                        <div class="form-title" id="formTitle">Add New Claim</div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label class="form-label">Source Type</label>
                                <select class="form-select" id="sourceType">
                                    <option value="paper">Paper</option>
                                    <option value="website">Website</option>
                                    <option value="book">Book</option>
                                    <option value="report">Report</option>
                                    <option value="other">Other</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Year</label>
                                <input type="text" class="form-input" id="claimYear" placeholder="e.g., 2023">
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label class="form-label">Claimed Content <small>(the factual claim)</small></label>
                            <textarea class="form-textarea" id="claimContent" placeholder="Enter the specific factual claim being made..."></textarea>
                        </div>
                        
                        <div class="form-group">
                            <label class="form-label">Title <small>(of the referenced source)</small></label>
                            <input type="text" class="form-input" id="claimTitle" placeholder="Title of the paper/website/book...">
                        </div>
                        
                        <div class="form-row">
                            <div class="form-group">
                                <label class="form-label">Authors</label>
                                <input type="text" class="form-input" id="claimAuthors" placeholder="e.g., Smith et al.">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Institution</label>
                                <input type="text" class="form-input" id="claimInstitution" placeholder="e.g., MIT, Nature">
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label class="form-label">URL</label>
                            <input type="text" class="form-input" id="claimUrl" placeholder="https://...">
                        </div>
                        
                        <div class="annotation-section">
                            <div class="annotation-title">Human Annotations</div>
                            
                            <div class="form-group">
                                <label class="form-label">Reference Grounding <small>(Is the reference real and retrievable?)</small></label>
                                <div class="options-grid" id="refOptions">
                                    <button type="button" class="option-btn yes" data-value="yes" onclick="selectOption('ref', 'yes', this)">Yes</button>
                                    <button type="button" class="option-btn no" data-value="no" onclick="selectOption('ref', 'no', this)">No</button>
                                    <button type="button" class="option-btn unsure" data-value="unsure" onclick="selectOption('ref', 'unsure', this)">Unsure</button>
                                </div>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Content Grounding <small>(Does the content match the source?)</small></label>
                                <div class="options-grid" id="contentOptions">
                                    <button type="button" class="option-btn yes" data-value="yes" onclick="selectOption('content', 'yes', this)">Yes</button>
                                    <button type="button" class="option-btn no" data-value="no" onclick="selectOption('content', 'no', this)">No</button>
                                    <button type="button" class="option-btn unsure" data-value="unsure" onclick="selectOption('content', 'unsure', this)">Unsure</button>
                                </div>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Comment <small>(optional notes)</small></label>
                                <textarea class="form-textarea" id="claimComment" style="min-height: 60px;" placeholder="Any additional notes..."></textarea>
                            </div>
                        </div>
                        
                        <div class="form-actions">
                            <button class="btn-clear" onclick="clearForm()">Clear Form</button>
                            <button class="btn-save" onclick="saveClaim()">Save Claim</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="shortcuts">
            <strong>Keyboard shortcuts:</strong>
            <kbd>←</kbd> Previous &nbsp;|&nbsp;
            <kbd>→</kbd> Next &nbsp;|&nbsp;
            <kbd>A</kbd> Add Claim &nbsp;|&nbsp;
            <kbd>1</kbd> Claims Tab &nbsp;|&nbsp;
            <kbd>2</kbd> Form Tab
        </div>
    </div>
    
    <div class="toast" id="toast">Saved!</div>
    
    <script>
        let editingIndex = -1;
        let refValue = '';
        let contentValue = '';
        
        function navigate(action) {{
            window.location.href = '/navigate?action=' + action;
        }}
        
        function jumpTo(num) {{
            const index = parseInt(num) - 1;
            if (index >= 0 && index < {total_count}) {{
                window.location.href = '/navigate?action=jump&index=' + index;
            }}
        }}
        
        function switchTab(tab) {{
            // Update tab buttons
            document.getElementById('tabClaims').classList.toggle('active', tab === 'claims');
            document.getElementById('tabForm').classList.toggle('active', tab === 'form');
            
            // Update tab content
            document.getElementById('tabContentClaims').classList.toggle('active', tab === 'claims');
            document.getElementById('tabContentForm').classList.toggle('active', tab === 'form');
        }}
        
        function openAddForm() {{
            editingIndex = -1;
            refValue = '';
            contentValue = '';
            
            // Reset form
            document.getElementById('sourceType').value = 'paper';
            document.getElementById('claimYear').value = '';
            document.getElementById('claimContent').value = '';
            document.getElementById('claimTitle').value = '';
            document.getElementById('claimAuthors').value = '';
            document.getElementById('claimInstitution').value = '';
            document.getElementById('claimUrl').value = '';
            document.getElementById('claimComment').value = '';
            
            // Reset option buttons
            document.querySelectorAll('.option-btn').forEach(btn => btn.classList.remove('selected'));
            
            // Update form title and tab label
            document.getElementById('formTitle').textContent = 'Add New Claim';
            document.getElementById('formTabLabel').textContent = 'Add Claim';
            
            // Switch to form tab
            switchTab('form');
        }}
        
        function editClaim(index) {{
            editingIndex = index;
            refValue = '';
            contentValue = '';
            
            // Reset form first
            document.getElementById('sourceType').value = 'paper';
            document.getElementById('claimYear').value = '';
            document.getElementById('claimContent').value = '';
            document.getElementById('claimTitle').value = '';
            document.getElementById('claimAuthors').value = '';
            document.getElementById('claimInstitution').value = '';
            document.getElementById('claimUrl').value = '';
            document.getElementById('claimComment').value = '';
            document.querySelectorAll('.option-btn').forEach(btn => btn.classList.remove('selected'));
            
            // Update form title and tab label
            document.getElementById('formTitle').textContent = 'Edit Claim #' + (index + 1);
            document.getElementById('formTabLabel').textContent = 'Edit #' + (index + 1);
            
            // Load existing data via API
            fetch('/get_claim?index=' + index)
                .then(r => r.json())
                .then(data => {{
                    document.getElementById('sourceType').value = data.inferred_source_type || 'paper';
                    document.getElementById('claimYear').value = data.claimed_year || '';
                    document.getElementById('claimContent').value = data.claimed_content || '';
                    document.getElementById('claimTitle').value = data.claimed_title || '';
                    document.getElementById('claimAuthors').value = data.claimed_authors || '';
                    document.getElementById('claimInstitution').value = data.claimed_institution || '';
                    document.getElementById('claimUrl').value = data.claimed_url || '';
                    document.getElementById('claimComment').value = data.human_comment || '';
                    
                    if (data.human_reference_grounding) {{
                        refValue = data.human_reference_grounding;
                        document.querySelector('#refOptions .option-btn.' + refValue).classList.add('selected');
                    }}
                    if (data.human_content_grounding) {{
                        contentValue = data.human_content_grounding;
                        document.querySelector('#contentOptions .option-btn.' + contentValue).classList.add('selected');
                    }}
                }});
            
            // Switch to form tab
            switchTab('form');
        }}
        
        function clearForm() {{
            editingIndex = -1;
            refValue = '';
            contentValue = '';
            
            document.getElementById('sourceType').value = 'paper';
            document.getElementById('claimYear').value = '';
            document.getElementById('claimContent').value = '';
            document.getElementById('claimTitle').value = '';
            document.getElementById('claimAuthors').value = '';
            document.getElementById('claimInstitution').value = '';
            document.getElementById('claimUrl').value = '';
            document.getElementById('claimComment').value = '';
            document.querySelectorAll('.option-btn').forEach(btn => btn.classList.remove('selected'));
            
            document.getElementById('formTitle').textContent = 'Add New Claim';
            document.getElementById('formTabLabel').textContent = 'Add Claim';
        }}
        
        function deleteClaim(index) {{
            if (confirm('Delete this claim?')) {{
                fetch('/delete_claim', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
                    body: 'index=' + index
                }})
                .then(r => r.text())
                .then(() => {{
                    window.location.href = window.location.pathname + '?t=' + Date.now();
                }});
            }}
        }}
        
        function selectOption(type, value, btn) {{
            const container = type === 'ref' ? document.getElementById('refOptions') : document.getElementById('contentOptions');
            container.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            
            if (type === 'ref') {{
                refValue = value;
            }} else {{
                contentValue = value;
            }}
        }}
        
        function saveClaim() {{
            const data = {{
                inferred_source_type: document.getElementById('sourceType').value,
                claimed_year: document.getElementById('claimYear').value,
                claimed_content: document.getElementById('claimContent').value,
                claimed_title: document.getElementById('claimTitle').value,
                claimed_authors: document.getElementById('claimAuthors').value,
                claimed_institution: document.getElementById('claimInstitution').value,
                claimed_url: document.getElementById('claimUrl').value,
                human_reference_grounding: refValue,
                human_content_grounding: contentValue,
                human_comment: document.getElementById('claimComment').value,
                editing_index: editingIndex
            }};
            
            if (!data.claimed_content.trim()) {{
                alert('Please enter the claimed content.');
                return;
            }}
            
            // Disable save button to prevent double-clicks
            const saveBtn = document.querySelector('.btn-save');
            saveBtn.disabled = true;
            saveBtn.textContent = 'Saving...';
            
            fetch('/save_claim', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(data)
            }})
            .then(r => r.text())  // Fully consume the response
            .then(text => {{
                showToast();
                // Use href assignment instead of reload for more reliable navigation
                window.location.href = window.location.pathname + '?t=' + Date.now();
            }})
            .catch(err => {{
                console.error('Save failed:', err);
                alert('Failed to save claim. Please try again.');
                saveBtn.disabled = false;
                saveBtn.textContent = 'Save Claim';
            }});
        }}
        
        function showToast() {{
            const toast = document.getElementById('toast');
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2000);
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            // Don't trigger if typing in input/textarea/select
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {{
                return;
            }}
            
            switch(e.key) {{
                case 'ArrowLeft':
                    if ({current} > 0) navigate('prev');
                    break;
                case 'ArrowRight':
                    if ({current} < {total_count - 1}) navigate('next');
                    break;
                case 'a':
                case 'A':
                    openAddForm();
                    break;
                case '1':
                    switchTab('claims');
                    break;
                case '2':
                    switchTab('form');
                    break;
            }}
        }});
    </script>
</body>
</html>'''

