# Claim Annotation Tool

A web-based tool for manually extracting and annotating claims from model responses.

## Quick Start

### Local Mode (Development)

Run locally and automatically open browser:

```bash
python annotate_responses.py conversations.jsonl
```

### Server Mode with File Picker (EC2/Remote)

Run as a web server with a file picker interface:

```bash
python annotate_responses.py --server-mode --data-dir /path/to/data --port 8080
```

Users can then:
1. Open the web interface in their browser
2. Select an input JSONL file from available files
3. Choose an existing output file or create a new one
4. Start annotating

### Server Mode with Password Protection (Recommended for EC2)

Add password protection to prevent unauthorized access:

```bash
python annotate_responses.py --server-mode --data-dir /path/to/data --password YOUR_SECRET
```

- Users must login before accessing the app
- Failed login attempts trigger a 5-second wait (brute-force protection)
- Sessions last 24 hours

### Server Mode with Pre-selected File

Skip the file picker by specifying files directly:

```bash
python annotate_responses.py conversations.jsonl --server-mode --port 8080
```

## Docker Deployment (Recommended for EC2)

### Build the Docker Image

```bash
docker build -t claim-annotator .
```

### Run with File Picker (Recommended)

```bash
docker run -d \
  --name claim-annotator \
  -p 8080:8080 \
  -v /path/to/your/data:/data \
  claim-annotator
```

Then open `http://your-ec2-ip:8080` and select files via the web interface.

### Run with Password Protection (Recommended for EC2!)

```bash
docker run -d \
  --name claim-annotator \
  -p 8080:8080 \
  -v /path/to/your/data:/data \
  claim-annotator --password YOUR_SECRET_PASSWORD
```

This requires users to login before accessing the app. Failed login attempts trigger a 5-second wait.

> **💾 Data Persistence**: The `-v /path/to/your/data:/data` volume mount stores all files 
> on your host machine, NOT inside the container. Your input files and annotation outputs 
> are preserved even when the container is stopped or deleted. Just restart the container 
> with the same volume mount to continue where you left off.

### Run with Pre-selected File

```bash
docker run -d \
  --name claim-annotator \
  -p 8080:8080 \
  -v /path/to/your/data:/data \
  claim-annotator /data/conversations.jsonl --output /data/annotations.jsonl
```

### Run with HTTPS (Recommended for Production)

First, generate SSL certificates (or use your own):

```bash
# Generate self-signed certificate for testing
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/CN=your-ec2-hostname"
```

Then run with SSL:

```bash
docker run -d \
  --name claim-annotator \
  -p 443:443 \
  -v /path/to/your/data:/data \
  -v /path/to/certs:/certs:ro \
  claim-annotator \
    --ssl-cert /certs/cert.pem \
    --ssl-key /certs/key.pem \
    --port 443
```

### Using Docker Compose

1. Place your JSONL files in `./data/`
2. Run:
   ```bash
   docker-compose up -d
   ```
3. Access at `http://your-ec2-ip:8080`
4. Select files via the web interface

## EC2 Setup Guide

### 1. Launch EC2 Instance

- **AMI**: Amazon Linux 2023 or Ubuntu 22.04
- **Instance type**: t3.micro (sufficient for annotation)
- **Security Group**: Allow inbound on port 8080 (or 443 for HTTPS)

### 2. Install Docker

```bash
# Amazon Linux 2023
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Ubuntu
sudo apt update
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

Log out and back in for group changes to take effect.

### 3. Deploy the Application

```bash
# Copy files to EC2
scp -i your-key.pem annotate_responses.py Dockerfile ubuntu@your-ec2-ip:~/

# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Build and run
cd ~
docker build -t claim-annotator .

# Create data directory and copy your files
mkdir data
# (upload your JSONL files to ~/data/)

# Run with file picker
docker run -d --name claim-annotator -p 8080:8080 -v ~/data:/data claim-annotator
```

### 4. Access the Tool

Navigate to `http://your-ec2-public-ip:8080` in your browser.

## Command Line Options

```
usage: annotate_responses.py [-h] [--output OUTPUT] [--port PORT] [--server-mode]
                              [--data-dir DATA_DIR] [--password PASSWORD]
                              [--host HOST] [--ssl-cert SSL_CERT] [--ssl-key SSL_KEY]
                              [--no-browser]
                              [input_file]

Arguments:
  input_file            Path to the conversations JSONL file 
                        (optional in server mode with --data-dir)

Options:
  --output OUTPUT       Path to output JSONL file (default: <input>_extracted.jsonl)
  --port PORT           Port to run the server on (default: 5050)
  --server-mode         Run in server mode (binds to 0.0.0.0, no browser auto-open)
  --data-dir DATA_DIR   Directory containing JSONL files (enables file picker)
  --password PASSWORD   Password to protect access (users must login first)
  --host HOST           Host to bind to (default: localhost or 0.0.0.0 in server mode)
  --ssl-cert SSL_CERT   Path to SSL certificate file for HTTPS
  --ssl-key SSL_KEY     Path to SSL private key file for HTTPS
  --no-browser          Don't automatically open browser (local mode only)
```

## File Picker Interface

When running in server mode with `--data-dir`, the tool presents a file picker interface:

1. **Available Files**: Shows all JSONL files in the data directory with metadata
2. **Input File Selection**: Click "Use as Input" to select the conversation file
3. **Output File**: Either select an existing file or enter a new filename
4. **Start Annotation**: Begin annotating once both are configured

Files are labeled as "input" or "output" based on naming conventions (`_extracted`, `_annotations`).

## Multi-User Support

The app supports **multiple simultaneous annotators** from different computers. Each user has:

- **Independent navigation** - Users can browse different parts of the file without affecting others
- **Independent file selection** - Each user can work on different files (in server mode with file picker)
- **Separate output files** - Each user's annotations are saved to their selected output file

### Example: Multiple Annotators on Same EC2

```bash
# Start server with file picker
docker run -d -p 8080:8080 -v ~/data:/data claim-annotator --password SECRET123

# User A opens http://ec2-ip:8080, selects input.jsonl → output_userA.jsonl
# User B opens http://ec2-ip:8080, selects input.jsonl → output_userB.jsonl

# Both users can annotate independently!
```

### Note on Shared Output Files

If two users select the **same output file**, the last save wins. For collaborative annotation, either:
- Have each user use a different output filename (e.g., `annotations_alice.jsonl`, `annotations_bob.jsonl`)
- Merge output files afterward

## Keyboard Shortcuts

- `←` Previous response
- `→` Next response
- `A` Add new claim
- `Esc` Close modal

## Output Format

The tool produces a JSONL file with the following structure:

```json
{
  "_type": "extraction_result",
  "conversation_id": 0,
  "turn_number": 1,
  "original_statement": "...",
  "extracted_claims": [
    {
      "inferred_source_type": "paper",
      "claimed_content": "...",
      "claimed_title": "...",
      "claimed_authors": "...",
      "claimed_year": "2023",
      "human_reference_grounding": "yes",
      "human_content_grounding": "no",
      "human_comment": "..."
    }
  ],
  "metadata": {...}
}
```
