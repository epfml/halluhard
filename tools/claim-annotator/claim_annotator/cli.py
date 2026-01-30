"""
Command-line interface and entry point.
"""

import argparse
import os
import ssl
import webbrowser
from http.server import ThreadingHTTPServer

from .state import STATE, hash_password
from .data import parse_conversations, load_existing_extractions, make_turn_key
from .handler import AnnotationHandler


def main():
    parser = argparse.ArgumentParser(
        description="Web-based claim extraction and annotation tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local mode (default) - opens browser automatically
  python annotate_responses.py conversations.jsonl

  # Server mode with file picker (for EC2 deployment)
  python annotate_responses.py --server-mode --data-dir /path/to/data --port 8080

  # Server mode with password protection
  python annotate_responses.py --server-mode --data-dir /data --password mysecretpass

  # Server mode with pre-selected file
  python annotate_responses.py conversations.jsonl --server-mode --port 8080

  # Server mode with HTTPS (recommended for production)
  python annotate_responses.py --server-mode --data-dir /data --ssl-cert cert.pem --ssl-key key.pem

  # Generate self-signed certificate for testing:
  openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
        """
    )
    parser.add_argument(
        "input_file", 
        nargs='?',  # Optional in server mode
        help="Path to the conversations JSONL file (optional in server mode with --data-dir)"
    )
    parser.add_argument("--output", help="Path to output JSONL file (default: <input_name>_extracted.jsonl)")
    parser.add_argument("--port", type=int, default=5050, help="Port to run the server on (default: 5050)")
    parser.add_argument(
        "--server-mode", 
        action="store_true",
        help="Run in server mode (binds to 0.0.0.0, no browser auto-open). Use for remote access like EC2."
    )
    parser.add_argument(
        "--data-dir",
        help="Directory containing JSONL files (enables file picker in server mode). Required if no input_file in server mode."
    )
    parser.add_argument(
        "--password",
        help="Password to protect access to the tool. Users must login before using the app."
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (default: 'localhost' for local mode, '0.0.0.0' for server mode)"
    )
    parser.add_argument(
        "--ssl-cert",
        help="Path to SSL certificate file (.pem) for HTTPS. Requires --ssl-key."
    )
    parser.add_argument(
        "--ssl-key",
        help="Path to SSL private key file (.pem) for HTTPS. Requires --ssl-cert."
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser (only applies to local mode)"
    )
    
    args = parser.parse_args()
    
    # Validate SSL arguments
    if (args.ssl_cert and not args.ssl_key) or (args.ssl_key and not args.ssl_cert):
        print("Error: Both --ssl-cert and --ssl-key must be provided together.")
        return
    
    if args.ssl_cert and not os.path.exists(args.ssl_cert):
        print(f"Error: SSL certificate file '{args.ssl_cert}' not found.")
        return
    
    if args.ssl_key and not os.path.exists(args.ssl_key):
        print(f"Error: SSL key file '{args.ssl_key}' not found.")
        return
    
    # Validate input file / data-dir requirements
    if not args.input_file and not args.data_dir:
        if args.server_mode:
            print("Error: In server mode, provide either an input_file or --data-dir for the file picker.")
        else:
            print("Error: Please provide an input file.")
        return
    
    if args.input_file and not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found.")
        return
    
    if args.data_dir and not os.path.isdir(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return
    
    # Set up password protection
    if args.password:
        STATE["password_hash"] = hash_password(args.password)
        print("🔐 Password protection enabled")
    else:
        STATE["password_hash"] = None
    
    # Determine host binding
    if args.host:
        host = args.host
    elif args.server_mode:
        host = '0.0.0.0'  # Bind to all interfaces for remote access
    else:
        host = 'localhost'  # Local-only binding
    
    # Set server mode state
    STATE["server_mode"] = args.server_mode
    STATE["data_dir"] = args.data_dir or (os.path.dirname(args.input_file) if args.input_file else ".")
    
    # If input file is provided, set up preload config
    if args.input_file:
        # Set default output path
        if args.output:
            output_path = args.output
        else:
            base_name = os.path.splitext(args.input_file)[0]
            output_path = f"{base_name}_extracted.jsonl"
        
        # Store preload config - will be loaded when sessions are created
        STATE["preload_input"] = args.input_file
        STATE["preload_output"] = output_path
        
        print(f"Input file: {args.input_file}")
        print(f"Output file: {output_path}")
        
        # Validate the file is parseable
        try:
            conversations, assistant_turns = parse_conversations(args.input_file)
            print(f"Found {len(conversations)} conversations with {len(assistant_turns)} assistant turns.")
            if not assistant_turns:
                print("No assistant turns found in conversations.")
                return
            # Check existing extractions
            existing = load_existing_extractions(output_path)
            annotated_count = sum(1 for turn in assistant_turns
                                 if make_turn_key(turn['conv_idx'], turn['turn_number']) in existing
                                 and len(existing[make_turn_key(turn['conv_idx'], turn['turn_number'])]) > 0)
            print(f"Found {annotated_count} existing annotated turns.")
        except Exception as e:
            print(f"Failed to parse input file: {e}")
            return
    else:
        # Server mode with file picker
        print(f"Data directory: {STATE['data_dir']}")
    
    # Determine protocol
    use_ssl = args.ssl_cert and args.ssl_key
    protocol = "https" if use_ssl else "http"
    
    # Start server (threaded to handle multiple concurrent requests)
    server = ThreadingHTTPServer((host, args.port), AnnotationHandler)
    
    # Wrap socket with SSL if certificates provided
    if use_ssl:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=args.ssl_cert, keyfile=args.ssl_key)
        server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
        print(f"SSL/TLS enabled with certificate: {args.ssl_cert}")
    
    # Construct URL for display
    display_host = 'localhost' if host == '0.0.0.0' else host
    url = f"{protocol}://{display_host}:{args.port}"
    
    print(f"\n{'='*60}")
    if args.server_mode:
        print(f"  🌐 SERVER MODE - Accessible from network")
        print(f"  Bound to: {host}:{args.port}")
        print(f"  Local URL: {url}")
        print(f"  Remote URL: {protocol}://<your-ip>:{args.port}")
        if not args.input_file:
            print(f"\n  📁 File picker enabled - select files via web interface")
        if args.password:
            print(f"  🔐 Password protection: ENABLED")
        else:
            print(f"  🔓 Password protection: DISABLED (open access)")
        if not use_ssl:
            print(f"\n  ⚠️  WARNING: Running without HTTPS!")
            print(f"  Consider using --ssl-cert and --ssl-key for production.")
    else:
        print(f"  🏠 LOCAL MODE")
        print(f"  Claim Extraction server running at: {url}")
        if args.password:
            print(f"  🔐 Password protection: ENABLED")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
    # Open browser only in local mode (unless --no-browser)
    if not args.server_mode and not args.no_browser:
        webbrowser.open(url)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        # Show output files from active sessions
        active_outputs = set()
        for session in STATE["sessions"].values():
            if session.output_path and session.file_loaded:
                active_outputs.add(session.output_path)
        if active_outputs:
            print("Extractions saved to:")
            for path in active_outputs:
                print(f"  - {path}")


if __name__ == "__main__":
    main()

