"""
Data parsing and file operations.
"""

import json
import os
import threading

_OUTPUT_LOCKS: dict[str, threading.Lock] = {}
_OUTPUT_LOCKS_GUARD = threading.Lock()


def _get_output_lock(output_path: str) -> threading.Lock:
    """Return a stable per-output-file lock to serialize appends."""
    with _OUTPUT_LOCKS_GUARD:
        lock = _OUTPUT_LOCKS.get(output_path)
        if lock is None:
            lock = threading.Lock()
            _OUTPUT_LOCKS[output_path] = lock
        return lock


def parse_conversations(file_path):
    """Parse conversation JSONL file and extract assistant turns."""
    conversations = []
    assistant_turns = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            conv_idx = len(conversations)
            conversations.append(data)
            
            conversation_id = data.get('conversation_id', line_num)
            turns = data.get('turns', [])
            metadata = data.get('metadata', {})
            
            for turn_idx, turn in enumerate(turns):
                if turn.get('role') == 'assistant':
                    assistant_turns.append({
                        "conv_idx": conv_idx,
                        "turn_idx": turn_idx,
                        "turn_number": turn.get('turn_index', turn_idx),
                        "content": turn.get('content', ''),
                        "conversation_id": conversation_id,
                        "metadata": metadata,
                        # Get the user question (previous turn if exists)
                        "user_question": turns[turn_idx - 1].get('content', '') if turn_idx > 0 else ''
                    })
    
    return conversations, assistant_turns


def make_turn_key(conv_idx, turn_number):
    """Create a consistent key for a turn using conv_idx (unique line index) and turn_number."""
    return f"{conv_idx}:{turn_number}"


def load_existing_extractions(output_path):
    """Load existing extractions from output file."""
    extractions = {}
    
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    conv_idx = data.get('conv_idx', None)
                    turn_num = data.get('turn_number', 0)
                    if conv_idx is not None:
                        key = make_turn_key(conv_idx, turn_num)
                        extractions[key] = data.get('extracted_claims', [])
                except json.JSONDecodeError:
                    continue
    
    return extractions


def get_jsonl_files(directory):
    """Get list of JSONL files in directory with metadata."""
    files = []
    if not os.path.isdir(directory):
        return files
    
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(directory, filename)
            try:
                stat = os.stat(filepath)
                # Count lines/entries
                with open(filepath, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for line in f if line.strip())
                files.append({
                    'name': filename,
                    'path': filepath,
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'entries': line_count
                })
            except Exception:
                continue
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x['modified'], reverse=True)
    return files


def load_file_for_annotation(session, input_path, output_path):
    """Load a file for annotation and update session state."""
    session.input_file = input_path
    session.output_path = output_path
    
    # Parse conversations
    session.conversations, session.assistant_turns = parse_conversations(input_path)
    
    if not session.assistant_turns:
        return False, "No assistant turns found in the file."
    
    # Load existing extractions
    session.extracted_claims = load_existing_extractions(output_path)
    
    # Find first unannotated turn
    session.current_flat_index = 0
    for i, turn in enumerate(session.assistant_turns):
        key = make_turn_key(turn['conv_idx'], turn['turn_number'])
        if key not in session.extracted_claims or len(session.extracted_claims[key]) == 0:
            session.current_flat_index = i
            break
    
    session.file_loaded = True
    return True, None


def save_all_extractions(session):
    """Save all extractions to output file for a specific session."""
    with open(session.output_path, 'w', encoding='utf-8') as f:
        for turn_data in session.assistant_turns:
            key = make_turn_key(turn_data['conv_idx'], turn_data['turn_number'])
            claims = session.extracted_claims.get(key, [])
            
            if claims:  # Only save if there are claims
                entry = {
                    "_type": "extraction_result",
                    "conv_idx": turn_data['conv_idx'],
                    "conversation_id": turn_data['conversation_id'],
                    "turn_number": turn_data['turn_number'],
                    "original_statement": turn_data['content'],
                    "extracted_claims": claims,
                    "metadata": turn_data['metadata']
                }
                f.write(json.dumps(entry) + "\n")


def save_extraction_update(session, turn_data):
    """
    Append-only save for a single turn.

    This is much faster than rewriting the whole output file on every save, and
    works well with multiple concurrent users.

    Note: We write a line even when `extracted_claims` is empty so that deletions
    (or clearing all claims for a turn) are reflected when reloading.
    """
    if not getattr(session, "output_path", None):
        return

    key = make_turn_key(turn_data["conv_idx"], turn_data["turn_number"])
    claims = session.extracted_claims.get(key, [])

    entry = {
        "_type": "extraction_result",
        "conv_idx": turn_data["conv_idx"],
        "conversation_id": turn_data.get("conversation_id"),
        "turn_number": turn_data["turn_number"],
        "original_statement": turn_data.get("content", ""),
        "extracted_claims": claims,
        "metadata": turn_data.get("metadata", {}),
    }

    lock = _get_output_lock(session.output_path)
    with lock:
        with open(session.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

