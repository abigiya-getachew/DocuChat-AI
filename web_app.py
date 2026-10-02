import sys
import os

# Ensure both relative and absolute src paths are on sys.path
sys.path.append("./src")
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from flask import Flask, render_template_string, request, jsonify
from src.document_loader import process_file
from src.embedding_engine import embed_documents
from src.vector_store import add_documents, collection, remove_documents_by_source
from src.llm_responder import answer_query
from src.config import Config

app = Flask(__name__)

def get_chunk_count():
    try:
        return collection.count()
    except Exception:
        return 0

def get_uploaded_documents():
    docs = []
    try:
        data = collection.get(include=["metadatas"])
        source_counts = {}
        if data and data.get("metadatas"):
            for meta in data["metadatas"]:
                if meta and "source" in meta:
                    src = meta["source"]
                    source_counts[src] = source_counts.get(src, 0) + 1

        for filename, doc_chunks in sorted(source_counts.items()):
            path = os.path.join(Config.DATA_FOLDER, filename)
            size_str = "N/A"
            if os.path.exists(path):
                try:
                    size_bytes = os.path.getsize(path)
                    size_mb = size_bytes / (1024 * 1024)
                    size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{size_bytes / 1024:.0f} KB"
                except Exception:
                    pass
            docs.append({
                "name": filename,
                "size": size_str,
                "chunks": doc_chunks,
                "indexed": True
            })
    except Exception as e:
        print(f"Error querying active documents: {e}")
    return docs


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocuChat AI — Clean Document Intelligence</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root, [data-theme="light"] {
            --bg-base: #F8FAFC;
            --bg-surface: #FFFFFF;
            --bg-card: rgba(255, 255, 255, 0.95);
            --bg-card-hover: #FFFFFF;
            --bg-input: #FFFFFF;
            
            --accent-primary: #2563EB;
            --accent-secondary: #1D4ED8;
            --accent-violet: #4F46E5;
            --accent-cyan: #0284C7;
            --accent-gradient: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
            --accent-glow: 0 4px 14px rgba(37, 99, 235, 0.25);
            
            --border-subtle: #E2E8F0;
            --border-highlight: rgba(37, 99, 235, 0.4);
            
            --text-main: #0F172A;
            --text-secondary: #475569;
            --text-muted: #94A3B8;
            --emerald-badge: #059669;
            --badge-bg: rgba(37, 99, 235, 0.08);
            --badge-border: rgba(37, 99, 235, 0.2);
            --badge-color: #2563EB;
            
            --header-bg: rgba(255, 255, 255, 0.85);
            --header-border: #E2E8F0;
            --shadow-card: 0 4px 20px -2px rgba(15, 23, 42, 0.06), 0 2px 6px -1px rgba(15, 23, 42, 0.04);
            --shadow-float: 0 10px 25px -5px rgba(15, 23, 42, 0.08);
            
            --dropzone-bg: rgba(37, 99, 235, 0.02);
            --dropzone-border: rgba(37, 99, 235, 0.25);
            --dropzone-hover-bg: rgba(37, 99, 235, 0.05);
            
            --msg-user-bg: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            --msg-user-color: #FFFFFF;
            --msg-bot-bg: #FFFFFF;
            --msg-bot-border: #E2E8F0;
            --msg-bot-color: #1E293B;
            
            --citation-bg: rgba(37, 99, 235, 0.08);
            --citation-border: rgba(37, 99, 235, 0.22);
            --citation-color: #1D4ED8;
            
            --empty-icon-bg: rgba(37, 99, 235, 0.08);
            --empty-icon-border: rgba(37, 99, 235, 0.2);
            --empty-icon-color: #2563EB;
            
            --chip-bg: #F1F5F9;
            --item-bg: #F8FAFC;
            --item-hover-bg: #F1F5F9;
            
            --radius-xl: 18px;
            --radius-lg: 12px;
            --radius-md: 8px;
            --radius-sm: 6px;
        }

        [data-theme="dark"] {
            --bg-base: #080C14;
            --bg-surface: #0E1424;
            --bg-card: rgba(18, 26, 44, 0.75);
            --bg-card-hover: rgba(25, 36, 61, 0.85);
            --bg-input: rgba(11, 17, 30, 0.85);
            
            --accent-primary: #6366F1;
            --accent-secondary: #4F46E5;
            --accent-violet: #8B5CF6;
            --accent-cyan: #06B6D4;
            --accent-gradient: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
            --accent-glow: 0 0 25px rgba(99, 102, 241, 0.35);
            
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(99, 102, 241, 0.4);
            
            --text-main: #F8FAFC;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;
            --emerald-badge: #10B981;
            --badge-bg: rgba(99, 102, 241, 0.15);
            --badge-border: rgba(99, 102, 241, 0.35);
            --badge-color: #A5B4FC;
            
            --header-bg: rgba(11, 16, 28, 0.7);
            --header-border: rgba(255, 255, 255, 0.08);
            --shadow-card: 0 10px 30px rgba(0, 0, 0, 0.25);
            --shadow-float: 0 12px 32px rgba(0, 0, 0, 0.35);
            
            --dropzone-bg: rgba(99, 102, 241, 0.03);
            --dropzone-border: rgba(99, 102, 241, 0.3);
            --dropzone-hover-bg: rgba(99, 102, 241, 0.08);
            
            --msg-user-bg: linear-gradient(135deg, #4338CA 0%, #3730A3 100%);
            --msg-user-color: #F8FAFC;
            --msg-bot-bg: rgba(25, 35, 58, 0.85);
            --msg-bot-border: rgba(255, 255, 255, 0.08);
            --msg-bot-color: #E2E8F0;
            
            --citation-bg: rgba(99, 102, 241, 0.15);
            --citation-border: rgba(99, 102, 241, 0.35);
            --citation-color: #C7D2FE;
            
            --empty-icon-bg: rgba(99, 102, 241, 0.12);
            --empty-icon-border: rgba(99, 102, 241, 0.25);
            --empty-icon-color: #818CF8;
            
            --chip-bg: rgba(255, 255, 255, 0.04);
            --item-bg: rgba(255, 255, 255, 0.03);
            --item-hover-bg: rgba(255, 255, 255, 0.06);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background-color 0.2s ease, border-color 0.2s ease, color 0.15s ease;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(37, 99, 235, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(79, 70, 229, 0.04) 0%, transparent 40%);
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 7px;
            height: 7px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.4);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(148, 163, 184, 0.7);
        }

        /* Header Bar */
        header {
            border-bottom: 1px solid var(--header-border);
            background: var(--header-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            position: sticky;
            top: 0;
            z-index: 100;
            padding: 14px 28px;
        }

        .header-content {
            max-width: 1440px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            background: var(--accent-gradient);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: var(--accent-glow);
            font-size: 20px;
            color: white;
        }

        .brand-title {
            font-family: 'Outfit', sans-serif;
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, var(--text-main) 30%, var(--accent-primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-badge {
            font-size: 11px;
            font-weight: 600;
            background: var(--badge-bg);
            border: 1px solid var(--badge-border);
            color: var(--badge-color);
            padding: 3px 8px;
            border-radius: 20px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .telemetry-bar {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .pill-badge {
            display: flex;
            align-items: center;
            gap: 7px;
            background: var(--chip-bg);
            border: 1px solid var(--border-subtle);
            padding: 6px 14px;
            border-radius: 30px;
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .pulse-indicator {
            width: 8px;
            height: 8px;
            background: var(--emerald-badge);
            border-radius: 50%;
            position: relative;
        }

        .pulse-indicator::after {
            content: '';
            position: absolute;
            top: -3px;
            left: -3px;
            right: -3px;
            bottom: -3px;
            border-radius: 50%;
            background: rgba(16, 185, 129, 0.35);
            animation: pulseGlow 2s infinite ease-in-out;
        }

        @keyframes pulseGlow {
            0%, 100% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.6); opacity: 0; }
        }

        .theme-toggle-btn {
            background: var(--chip-bg);
            border: 1px solid var(--border-subtle);
            color: var(--text-main);
            padding: 6px 12px;
            border-radius: 30px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        }

        .theme-toggle-btn:hover {
            border-color: var(--accent-primary);
            color: var(--accent-primary);
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12);
        }

        /* Main Workspace Layout */
        .workspace {
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px 28px;
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 24px;
            flex: 1;
            width: 100%;
        }

        @media (max-width: 960px) {
            .workspace {
                grid-template-columns: 1fr;
            }
        }

        /* Sidebar Panels */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .panel-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xl);
            padding: 22px;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: var(--shadow-card);
            transition: all 0.25s ease;
        }

        .panel-card:hover {
            border-color: var(--border-highlight);
            box-shadow: var(--shadow-float);
        }

        .panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }

        .panel-title {
            font-family: 'Outfit', sans-serif;
            font-size: 16px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-main);
        }

        /* Upload Area */
        .upload-dropzone {
            border: 2px dashed var(--dropzone-border);
            border-radius: var(--radius-lg);
            padding: 28px 18px;
            text-align: center;
            background: var(--dropzone-bg);
            cursor: pointer;
            transition: all 0.25s ease;
            position: relative;
        }

        .upload-dropzone:hover, .upload-dropzone.drag-active {
            border-color: var(--accent-primary);
            background: var(--dropzone-hover-bg);
            transform: translateY(-1px);
        }

        .upload-dropzone input[type="file"] {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }

        .upload-icon {
            width: 44px;
            height: 44px;
            border-radius: 12px;
            background: var(--badge-bg);
            color: var(--accent-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 12px;
            transition: transform 0.25s ease;
        }

        .upload-dropzone:hover .upload-icon {
            transform: scale(1.08);
        }

        .upload-text-primary {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 4px;
        }

        .upload-text-secondary {
            font-size: 12px;
            color: var(--text-muted);
        }

        .selected-file-chip {
            display: none;
            align-items: center;
            justify-content: space-between;
            background: var(--chip-bg);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 10px 12px;
            margin-top: 14px;
            font-size: 13px;
        }

        .file-info {
            display: flex;
            align-items: center;
            gap: 8px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .action-button {
            width: 100%;
            background: var(--accent-gradient);
            color: white;
            border: none;
            border-radius: var(--radius-md);
            padding: 13px 18px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.25s ease;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
        }

        .action-button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
            filter: brightness(1.05);
        }

        .action-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        /* Progress Bar */
        .progress-box {
            display: none;
            margin-top: 14px;
            padding: 14px;
            background: var(--badge-bg);
            border: 1px solid var(--badge-border);
            border-radius: var(--radius-md);
        }

        .progress-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            font-weight: 500;
            color: var(--accent-primary);
            margin-bottom: 8px;
        }

        .progress-bar-bg {
            height: 6px;
            background: rgba(0, 0, 0, 0.08);
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }

        .progress-bar-fill {
            height: 100%;
            width: 100%;
            background: var(--accent-gradient);
            animation: indeterminate 1.8s infinite ease-in-out;
            transform-origin: 0% 50%;
        }

        @keyframes indeterminate {
            0% { transform: translateX(-100%); }
            50% { transform: translateX(0%); }
            100% { transform: translateX(100%); }
        }

        /* Document Library List */
        .doc-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-height: 240px;
            overflow-y: auto;
            margin-top: 8px;
        }

        .doc-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--item-bg);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 10px 14px;
            transition: all 0.2s ease;
        }

        .doc-item:hover {
            background: var(--item-hover-bg);
            border-color: var(--border-highlight);
        }

        .doc-meta {
            display: flex;
            align-items: center;
            gap: 10px;
            overflow: hidden;
        }

        .doc-icon {
            color: #DC2626;
            font-size: 18px;
            flex-shrink: 0;
        }

        .doc-name {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-main);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .doc-size {
            font-size: 11px;
            color: var(--text-muted);
        }

        .doc-status-badge {
            font-size: 11px;
            font-weight: 600;
            color: var(--emerald-badge);
            background: rgba(16, 185, 129, 0.12);
            padding: 3px 8px;
            border-radius: 12px;
            flex-shrink: 0;
        }

        .doc-item-actions {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-shrink: 0;
        }

        .btn-delete-doc {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            cursor: pointer;
            padding: 4px 6px;
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }

        .btn-delete-doc:hover {
            color: #EF4444;
            background: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.25);
        }


        /* Suggestions Pills */
        .suggestion-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 6px;
        }

        .suggestion-pill {
            background: var(--item-bg);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 10px 14px;
            font-size: 12px;
            color: var(--text-secondary);
            cursor: pointer;
            text-align: left;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .suggestion-pill:hover {
            background: var(--badge-bg);
            border-color: var(--badge-border);
            color: var(--accent-primary);
            transform: translateX(3px);
        }

        /* Right Side: Chat Studio */
        .chat-studio {
            display: flex;
            flex-direction: column;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-xl);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: var(--shadow-card);
            height: calc(100vh - 120px);
            min-height: 600px;
            overflow: hidden;
        }

        .chat-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-surface);
        }

        .chat-title-wrap {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .chat-avatar-badge {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: var(--badge-bg);
            color: var(--accent-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }

        .chat-title {
            font-family: 'Outfit', sans-serif;
            font-size: 16px;
            font-weight: 600;
            color: var(--text-main);
        }

        .chat-subtitle {
            font-size: 12px;
            color: var(--text-muted);
        }

        .btn-ghost {
            background: transparent;
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            border-radius: var(--radius-sm);
            padding: 6px 12px;
            font-size: 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
        }

        .btn-ghost:hover {
            background: var(--chip-bg);
            color: var(--text-main);
            border-color: var(--accent-primary);
        }

        /* Messages Stream */
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* Empty State */
        .chat-empty {
            margin: auto;
            text-align: center;
            max-width: 440px;
            padding: 20px;
        }

        .empty-icon {
            width: 64px;
            height: 64px;
            border-radius: 20px;
            background: var(--empty-icon-bg);
            color: var(--empty-icon-color);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 30px;
            margin: 0 auto 16px;
            border: 1px solid var(--empty-icon-border);
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.1);
        }

        .empty-title {
            font-family: 'Outfit', sans-serif;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 8px;
            color: var(--text-main);
        }

        .empty-desc {
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        /* Message Bubbles */
        .msg-row {
            display: flex;
            gap: 14px;
            max-width: 88%;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .msg-row.user {
            margin-left: auto;
            flex-direction: row-reverse;
        }

        .msg-avatar {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 600;
            flex-shrink: 0;
        }

        .msg-row.user .msg-avatar {
            background: var(--accent-gradient);
            color: white;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
        }

        .msg-row.bot .msg-avatar {
            background: var(--badge-bg);
            color: var(--accent-primary);
            border: 1px solid var(--badge-border);
        }

        .msg-content {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .msg-bubble {
            padding: 14px 18px;
            border-radius: var(--radius-lg);
            font-size: 14px;
            line-height: 1.65;
            position: relative;
        }

        .msg-row.user .msg-bubble {
            background: var(--msg-user-bg);
            color: var(--msg-user-color);
            border-bottom-right-radius: 4px;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.2);
        }

        .msg-row.bot .msg-bubble {
            background: var(--msg-bot-bg);
            border: 1px solid var(--msg-bot-border);
            color: var(--msg-bot-color);
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        }

        .msg-actions {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 4px;
        }

        .btn-copy {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 11px;
            display: flex;
            align-items: center;
            gap: 4px;
            cursor: pointer;
            padding: 2px 6px;
            border-radius: 4px;
            transition: all 0.2s ease;
        }

        .btn-copy:hover {
            color: var(--accent-primary);
            background: var(--chip-bg);
        }

        /* Citation Badge */
        .citation-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--citation-bg);
            border: 1px solid var(--citation-border);
            color: var(--citation-color);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            margin: 6px 0;
        }

        /* Thinking / Loading Animation */
        .thinking-bubble {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 18px;
            background: var(--msg-bot-bg);
            border: 1px solid var(--badge-border);
            border-radius: var(--radius-lg);
            border-bottom-left-radius: 4px;
            color: var(--accent-primary);
            font-size: 13px;
        }

        .typing-dots {
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .typing-dots span {
            width: 6px;
            height: 6px;
            background: var(--accent-primary);
            border-radius: 50%;
            animation: bounceDot 1.4s infinite ease-in-out both;
        }

        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounceDot {
            0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
            40% { transform: scale(1); opacity: 1; }
        }

        /* Chat Input Dock */
        .chat-input-bar {
            padding: 16px 24px;
            background: var(--bg-surface);
            border-top: 1px solid var(--border-subtle);
        }

        .input-form {
            display: flex;
            align-items: center;
            gap: 12px;
            background: var(--bg-input);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 6px 8px 6px 16px;
            transition: all 0.25s ease;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        }

        .input-form:focus-within {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
        }

        .chat-input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-main);
            font-family: inherit;
            font-size: 14px;
            padding: 8px 0;
        }

        .chat-input::placeholder {
            color: var(--text-muted);
        }

        .send-button {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: var(--accent-gradient);
            border: none;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
            flex-shrink: 0;
        }

        .send-button:hover:not(:disabled) {
            transform: scale(1.05);
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        }

        .send-button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }

        /* Toast Alert */
        .toast-container {
            position: fixed;
            bottom: 24px;
            right: 24px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .toast {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 12px 18px;
            font-size: 13px;
            color: var(--text-main);
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideInRight 0.3s ease;
            backdrop-filter: blur(16px);
        }

        .toast.success { border-color: rgba(16, 185, 129, 0.5); }
        .toast.error { border-color: rgba(239, 68, 68, 0.5); }

        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header>
        <div class="header-content">
            <div class="brand-container">
                <div class="brand-icon">⚡</div>
                <div>
                    <span class="brand-title">DocuChat AI</span>
                </div>
                <span class="brand-badge">RAG Studio</span>
            </div>

            <div class="telemetry-bar">
                <div class="pill-badge">
                    <span class="pulse-indicator"></span>
                    <span>Ollama: <strong>{{ config.DEFAULT_MODEL }}</strong></span>
                </div>
                <div class="pill-badge" id="header-chunk-badge">
                    <span>📚</span>
                    <span><strong id="chunk-counter">{{ chunk_count }}</strong> Chunks</span>
                </div>
                <button class="theme-toggle-btn" onclick="toggleTheme()" id="theme-btn" title="Toggle Light / Dark mode">
                    <span id="theme-icon">🌙</span>
                    <span id="theme-text">Dark</span>
                </button>
            </div>
        </div>
    </header>

    <!-- Main Workspace -->
    <main class="workspace">
        
        <!-- Left Sidebar: Documents -->
        <aside class="sidebar">
            
            <!-- Upload Panel -->
            <div class="panel-card">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>📥</span> Upload Document
                    </div>
                    <span class="brand-badge">PDF only</span>
                </div>

                <form id="upload-form" method="post" enctype="multipart/form-data" action="/upload" onsubmit="handleUpload(event)">
                    <div class="upload-dropzone" id="dropzone">
                        <input type="file" id="file-input" name="file" accept=".pdf" required onchange="handleFileSelected(this)">
                        <div class="upload-icon">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                <polyline points="17 8 12 3 7 8"></polyline>
                                <line x1="12" y1="3" x2="12" y2="15"></line>
                            </svg>
                        </div>
                        <div class="upload-text-primary">Choose PDF or drop here</div>
                        <div class="upload-text-secondary">Chunking & Vectorization automatic</div>
                    </div>

                    <div class="selected-file-chip" id="selected-file-chip">
                        <div class="file-info">
                            <span>📄</span>
                            <span id="selected-filename">filename.pdf</span>
                        </div>
                        <span id="selected-filesize" style="color: var(--text-muted); font-size: 11px;">0 KB</span>
                    </div>

                    <div class="progress-box" id="upload-progress">
                        <div class="progress-header">
                            <span id="progress-status-text">Embedding & indexing chunks...</span>
                            <span>⚡ Ollama Active</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill"></div>
                        </div>
                    </div>

                    <button type="submit" id="btn-upload" class="action-button">
                        <span>Index Document</span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="5" y1="12" x2="19" y2="12"></line>
                            <polyline points="12 5 19 12 12 19"></polyline>
                        </svg>
                    </button>
                </form>
            </div>

            <!-- Indexed Documents Panel -->
            <div class="panel-card">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>📑</span> Document Library
                    </div>
                    <span class="pill-badge" style="padding: 2px 8px; font-size: 11px;" id="doc-count-badge">{{ documents|length }} files</span>
                </div>

                <div class="doc-list" id="doc-list">
                    {% if documents %}
                        {% for doc in documents %}
                        <div class="doc-item">
                            <div class="doc-meta">
                                <span class="doc-icon">📄</span>
                                <div>
                                    <div class="doc-name" title="{{ doc.name }}">{{ doc.name }}</div>
                                    <div class="doc-size">{{ doc.size }} • {{ doc.chunks }} chunks</div>
                                </div>
                            </div>
                            <div class="doc-item-actions">
                                {% if doc.indexed %}
                                <span class="doc-status-badge">Indexed</span>
                                {% else %}
                                <span class="doc-status-badge" style="background: rgba(245, 158, 11, 0.12); color: #D97706;">Pending</span>
                                {% endif %}
                                <button class="btn-delete-doc" onclick="removeDocument('{{ doc.name }}', event)" title="Remove from workspace (local file is kept safe)">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <polyline points="3 6 5 6 21 6"></polyline>
                                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                                    </svg>
                                </button>
                            </div>
                        </div>
                        {% endfor %}
                    {% else %}
                        <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 16px;" id="no-docs-msg">
                            No documents uploaded yet. Upload a PDF above to begin.
                        </div>
                    {% endif %}
                </div>
            </div>

            <!-- Quick Suggestions -->
            <div class="panel-card">
                <div class="panel-header">
                    <div class="panel-title">
                        <span>💡</span> Prompt Starters
                    </div>
                </div>
                <div class="suggestion-group">
                    <div class="suggestion-pill" onclick="useSuggestion('Summarize the primary purpose and key points of this document.')">
                        <span>📌</span> Summarize the key points
                    </div>
                    <div class="suggestion-pill" onclick="useSuggestion('What are the core technical concepts explained in the text?')">
                        <span>🔍</span> Core technical concepts
                    </div>
                    <div class="suggestion-pill" onclick="useSuggestion('List any prerequisites, requirements, or steps mentioned.')">
                        <span>📋</span> List requirements & steps
                    </div>
                </div>
            </div>

        </aside>

        <!-- Right Side: Interactive Chat Studio -->
        <section class="chat-studio">
            <div class="chat-header">
                <div class="chat-title-wrap">
                    <div class="chat-avatar-badge">🤖</div>
                    <div>
                        <div class="chat-title">AI Assistant</div>
                        <div class="chat-subtitle">Grounded Q&A with Strict Document Citations</div>
                    </div>
                </div>

                <button class="btn-ghost" onclick="clearChat()">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                    Clear Chat
                </button>
            </div>

            <!-- Messages Stream -->
            <div class="chat-messages" id="chat-messages">
                
                {% if not question and not answer %}
                <div class="chat-empty" id="chat-empty">
                    <div class="empty-icon">✨</div>
                    <h3 class="empty-title">Ask anything from your documents</h3>
                    <p class="empty-desc">
                        DocuChat AI queries your Chroma vector store, retrieves high-relevance chunks, and generates precise answers with exact page citations.
                    </p>
                </div>
                {% endif %}

                {% if question %}
                <div class="msg-row user">
                    <div class="msg-avatar">U</div>
                    <div class="msg-content">
                        <div class="msg-bubble">{{ question }}</div>
                    </div>
                </div>
                {% endif %}

                {% if answer %}
                <div class="msg-row bot">
                    <div class="msg-avatar">🤖</div>
                    <div class="msg-content">
                        <div class="msg-bubble" id="initial-answer">{{ answer }}</div>
                        <div class="msg-actions">
                            <button class="btn-copy" onclick="copyText(this)">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                </svg>
                                Copy
                            </button>
                        </div>
                    </div>
                </div>
                {% endif %}

            </div>

            <!-- Input Dock -->
            <div class="chat-input-bar">
                <form id="ask-form" class="input-form" method="post" action="/ask" onsubmit="handleAsk(event)">
                    <input 
                        type="text" 
                        name="question" 
                        id="question-input" 
                        class="chat-input" 
                        placeholder="Ask a question about your indexed documents... (e.g. 'What is closure in JavaScript?')" 
                        autocomplete="off"
                        required
                    >
                    <button type="submit" id="btn-send" class="send-button" title="Send question">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="22" y1="2" x2="11" y2="13"></line>
                            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                        </svg>
                    </button>
                </form>
            </div>
        </section>

    </main>

    <!-- Toast Notifications -->
    <div class="toast-container" id="toast-container"></div>

    <script>
        // Theme Management (Light by default with persistence)
        function initTheme() {
            const savedTheme = localStorage.getItem('docuchat_theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            updateThemeButton(savedTheme);
        }

        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme') || 'light';
            const next = current === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('docuchat_theme', next);
            updateThemeButton(next);
        }

        function updateThemeButton(theme) {
            const icon = document.getElementById('theme-icon');
            const text = document.getElementById('theme-text');
            if (theme === 'light') {
                icon.textContent = '🌙';
                text.textContent = 'Dark';
            } else {
                icon.textContent = '☀️';
                text.textContent = 'Light';
            }
        }

        initTheme();

        // Formatting helper for citations and markdown
        function formatAIResponse(text) {
            if (!text) return '';

            let formatted = escapeHtml(text);

            // Format [Source: filename, Page X] and (Source: filename, Page X)
            formatted = formatted.replace(/(?:\\[|\\()Source:\\s*([^,)\\]]+),\\s*Page\\s*(\\d+)(?:\\]|\\))/gi, function(match, src, page) {
                return `<div class="citation-tag">📄 <strong>${src}</strong> • Page ${page}</div>`;
            });

            // Convert markdown headings to styled spans (prevents huge browser heading rendering)
            formatted = formatted.replace(/^#{6}\s+(.*)$/gm, '<span style="font-weight:600;font-size:13px;color:var(--text-primary);">$1</span>');
            formatted = formatted.replace(/^#{5}\s+(.*)$/gm, '<span style="font-weight:600;font-size:13px;color:var(--text-primary);">$1</span>');
            formatted = formatted.replace(/^#{4}\s+(.*)$/gm, '<span style="font-weight:600;font-size:14px;color:var(--text-primary);">$1</span>');
            formatted = formatted.replace(/^#{3}\s+(.*)$/gm, '<span style="font-weight:700;font-size:14px;color:var(--accent-primary);">$1</span>');
            formatted = formatted.replace(/^#{2}\s+(.*)$/gm, '<span style="font-weight:700;font-size:15px;color:var(--accent-primary);">$1</span>');
            formatted = formatted.replace(/^#\s+(.*)$/gm, '<span style="font-weight:700;font-size:15px;color:var(--accent-primary);">$1</span>');

            // Format bold/italic
            formatted = formatted.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
            formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');

            // Format inline code
            formatted = formatted.replace(/`([^`]+)`/g, '<code style="background:rgba(37,99,235,0.08);color:var(--accent-secondary);padding:2px 6px;border-radius:4px;font-family:monospace;font-size:12px;">$1</code>');

            // Format bullet points
            formatted = formatted.replace(/^\s*[\*\-]\s+(.*)$/gm, '<span style="display:block;padding-left:14px;">• $1</span>');

            // Format line breaks
            formatted = formatted.replace(/\\n/g, '<br>');
            return formatted;
        }

        // Apply formatting to initial answer if present
        document.addEventListener('DOMContentLoaded', () => {
            const initialAns = document.getElementById('initial-answer');
            if (initialAns) {
                initialAns.innerHTML = formatAIResponse(initialAns.innerText);
            }
        });

        // Toast Helper
        function showToast(message, type = 'info') {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.innerHTML = `<span>${type === 'success' ? '✅' : type === 'error' ? '⚠️' : 'ℹ️'}</span> <span>${message}</span>`;
            container.appendChild(toast);
            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transform = 'translateX(100%)';
                toast.style.transition = 'all 0.3s ease';
                setTimeout(() => toast.remove(), 300);
            }, 4000);
        }

        // File Selection Display
        function handleFileSelected(input) {
            const chip = document.getElementById('selected-file-chip');
            const nameEl = document.getElementById('selected-filename');
            const sizeEl = document.getElementById('selected-filesize');
            
            if (input.files && input.files[0]) {
                const file = input.files[0];
                nameEl.textContent = file.name;
                const sizeKb = (file.size / 1024).toFixed(0);
                sizeEl.textContent = sizeKb >= 1024 ? `${(sizeKb / 1024).toFixed(1)} MB` : `${sizeKb} KB`;
                chip.style.display = 'flex';
            } else {
                chip.style.display = 'none';
            }
        }

        // Drag and Drop Effects
        const dropzone = document.getElementById('dropzone');
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.add('drag-active');
            });
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.remove('drag-active');
            });
        });

        // AJAX Document Upload
        async function handleUpload(e) {
            e.preventDefault();
            const form = document.getElementById('upload-form');
            const fileInput = document.getElementById('file-input');
            const btnUpload = document.getElementById('btn-upload');
            const progress = document.getElementById('upload-progress');

            if (!fileInput.files || !fileInput.files[0]) {
                showToast("Please choose a PDF file to index.", "error");
                return;
            }

            const formData = new FormData(form);
            btnUpload.disabled = true;
            btnUpload.innerHTML = `<span>Processing...</span>`;
            progress.style.display = 'block';

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    headers: { 'Accept': 'application/json' },
                    body: formData
                });
                const result = await response.json();

                if (response.ok && result.success) {
                    showToast(result.message || "Document indexed successfully!", "success");
                    
                    // Update Chunk Counter
                    if (result.total_chunks !== undefined) {
                        document.getElementById('chunk-counter').textContent = result.total_chunks;
                    }
                    
                    // Update Document List
                    if (result.documents) {
                        updateDocumentList(result.documents);
                    }

                    // Reset File Input
                    form.reset();
                    document.getElementById('selected-file-chip').style.display = 'none';
                } else {
                    showToast(result.error || "Failed to index document.", "error");
                }
            } catch (err) {
                showToast("Connection error while indexing: " + err.message, "error");
            } finally {
                btnUpload.disabled = false;
                btnUpload.innerHTML = `<span>Index Document</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>`;
                progress.style.display = 'none';
            }
        }

        function updateDocumentList(docs) {
            const list = document.getElementById('doc-list');
            const countBadge = document.getElementById('doc-count-badge');
            countBadge.textContent = `${docs.length} files`;

            if (docs.length === 0) {
                list.innerHTML = `<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 16px;">No documents uploaded yet.</div>`;
                return;
            }

            list.innerHTML = docs.map(d => `
                <div class="doc-item">
                    <div class="doc-meta">
                        <span class="doc-icon">📄</span>
                        <div>
                            <div class="doc-name" title="${d.name}">${d.name}</div>
                            <div class="doc-size">${d.size} • ${d.chunks || 0} chunks</div>
                        </div>
                    </div>
                    <div class="doc-item-actions">
                        <span class="doc-status-badge" style="${d.indexed ? '' : 'background:rgba(245,158,11,0.12);color:#D97706;'}">
                            ${d.indexed ? 'Indexed' : 'Pending'}
                        </span>
                        <button class="btn-delete-doc" onclick="removeDocument('${d.name}', event)" title="Remove from workspace (local file is kept safe)">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3 6 5 6 21 6"></polyline>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            </svg>
                        </button>
                    </div>
                </div>
            `).join('');
        }

        async function removeDocument(filename, e) {
            if (e) e.stopPropagation();
            if (!confirm(`Remove "${filename}" from the AI workspace? This will remove its vectors from search, but your original file will remain safe in your folder.`)) {
                return;
            }
            try {
                const response = await fetch('/delete_document', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                    body: JSON.stringify({ filename })
                });
                const result = await response.json();
                if (response.ok && result.success) {
                    showToast(result.message, "success");
                    document.getElementById('chunk-counter').textContent = result.total_chunks;
                    updateDocumentList(result.documents);
                } else {
                    showToast(result.error || "Failed to remove document.", "error");
                }
            } catch (err) {
                showToast("Error deleting document: " + err.message, "error");
            }
        }


        // AJAX Question Asking
        async function handleAsk(e) {
            e.preventDefault();
            const input = document.getElementById('question-input');
            const question = input.value.trim();
            const messagesContainer = document.getElementById('chat-messages');
            const btnSend = document.getElementById('btn-send');
            const emptyState = document.getElementById('chat-empty');

            if (!question) return;
            if (emptyState) emptyState.remove();

            // Append User Bubble
            const userRow = document.createElement('div');
            userRow.className = 'msg-row user';
            userRow.innerHTML = `
                <div class="msg-avatar">U</div>
                <div class="msg-content">
                    <div class="msg-bubble">${escapeHtml(question)}</div>
                </div>
            `;
            messagesContainer.appendChild(userRow);
            input.value = '';
            btnSend.disabled = true;

            // Append Thinking Indicator
            const thinkingRow = document.createElement('div');
            thinkingRow.className = 'msg-row bot';
            thinkingRow.id = 'thinking-indicator';
            thinkingRow.innerHTML = `
                <div class="msg-avatar">🤖</div>
                <div class="msg-content">
                    <div class="thinking-bubble">
                        <div class="typing-dots">
                            <span></span><span></span><span></span>
                        </div>
                        <span>Searching document chunks & generating answer...</span>
                    </div>
                </div>
            `;
            messagesContainer.appendChild(thinkingRow);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({ question })
                });

                const result = await response.json();
                thinkingRow.remove();

                if (response.ok && result.success) {
                    const botRow = document.createElement('div');
                    botRow.className = 'msg-row bot';
                    const formattedAns = formatAIResponse(result.answer);
                    botRow.innerHTML = `
                        <div class="msg-avatar">🤖</div>
                        <div class="msg-content">
                            <div class="msg-bubble">${formattedAns}</div>
                            <div class="msg-actions">
                                <button class="btn-copy" onclick="copyText(this)">
                                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                    </svg>
                                    Copy
                                </button>
                            </div>
                        </div>
                    `;
                    messagesContainer.appendChild(botRow);
                } else {
                    const errRow = document.createElement('div');
                    errRow.className = 'msg-row bot';
                    errRow.innerHTML = `
                        <div class="msg-avatar" style="background:#FEE2E2; color:#DC2626;">!</div>
                        <div class="msg-content">
                            <div class="msg-bubble" style="border-color: #FCA5A5; color: #DC2626; background: #FFF5F5;">
                                ${escapeHtml(result.error || "Failed to generate answer.")}
                            </div>
                        </div>
                    `;
                    messagesContainer.appendChild(errRow);
                }
            } catch (err) {
                thinkingRow.remove();
                showToast("Request failed: " + err.message, "error");
            } finally {
                btnSend.disabled = false;
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                input.focus();
            }
        }

        // Suggestions — fill input and auto-submit
        function useSuggestion(text) {
            const input = document.getElementById('question-input');
            input.value = text;
            input.focus();
            // Auto-submit after a brief delay so user sees what was typed
            setTimeout(() => {
                const form = document.getElementById('ask-form');
                form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
            }, 120);
        }

        // Copy Text
        function copyText(btn) {
            const bubble = btn.closest('.msg-content').querySelector('.msg-bubble');
            const textToCopy = bubble.innerText || bubble.textContent;
            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalHtml = btn.innerHTML;
                btn.innerHTML = `<span>✓</span> Copied!`;
                btn.style.color = '#059669';
                setTimeout(() => {
                    btn.innerHTML = originalHtml;
                    btn.style.color = '';
                }, 2000);
            });
        }

        // Clear Chat
        function clearChat() {
            if (!confirm('Clear this conversation?')) return;
            const messages = document.getElementById('chat-messages');
            messages.innerHTML = `
                <div class="chat-empty" id="chat-empty">
                    <div class="empty-icon">✨</div>
                    <h3 class="empty-title">Ask anything from your documents</h3>
                    <p class="empty-desc">DocuChat AI retrieves high-relevance chunks and generates precise answers with exact page citations.</p>
                    <div style="display:flex;flex-direction:column;gap:8px;margin-top:16px;width:100%;max-width:380px;">
                        <div class="suggestion-pill" onclick="useSuggestion('Summarize the primary purpose and key points of this document.')">
                            <span>📌</span> Summarize the key points
                        </div>
                        <div class="suggestion-pill" onclick="useSuggestion('What are the core technical concepts explained in the text?')">
                            <span>🔍</span> Core technical concepts
                        </div>
                        <div class="suggestion-pill" onclick="useSuggestion('List any prerequisites, requirements, or steps mentioned.')">
                            <span>📋</span> List requirements &amp; steps
                        </div>
                    </div>
                </div>
            `;
            showToast('Chat cleared', 'success');
        }

        function escapeHtml(string) {
            const entityMap = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;'
            };
            return String(string).replace(/[&<>"']/g, function (s) {
                return entityMap[s];
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    chunk_count = get_chunk_count()
    docs = get_uploaded_documents()
    is_indexed = chunk_count > 0 or len(docs) > 0
    return render_template_string(
        HTML_TEMPLATE,
        indexed=is_indexed,
        chunk_count=chunk_count,
        documents=docs,
        config=Config
    )

@app.route('/upload', methods=['POST'])
def upload():
    is_ajax = request.headers.get('Accept') == 'application/json' or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json
    
    if 'file' not in request.files:
        if is_ajax:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        return render_template_string(HTML_TEMPLATE, error="No file uploaded", indexed=get_chunk_count() > 0, chunk_count=get_chunk_count(), documents=get_uploaded_documents(), config=Config), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        if is_ajax:
            return jsonify({"success": False, "error": "Please select a valid PDF file"}), 400
        return render_template_string(HTML_TEMPLATE, error="Please select a valid PDF file", indexed=get_chunk_count() > 0, chunk_count=get_chunk_count(), documents=get_uploaded_documents(), config=Config), 400

    try:
        os.makedirs(Config.DATA_FOLDER, exist_ok=True)
        file_path = os.path.join(Config.DATA_FOLDER, file.filename)
        file.save(file_path)

        chunks = process_file(file.filename)
        embedded = embed_documents(chunks)
        add_documents(embedded)

        total_chunks = get_chunk_count()
        docs = get_uploaded_documents()

        if is_ajax:
            return jsonify({
                "success": True,
                "filename": file.filename,
                "chunks_added": len(chunks),
                "total_chunks": total_chunks,
                "documents": docs,
                "message": f"Successfully indexed {file.filename} ({len(chunks)} chunks)!"
            })
        
        return render_template_string(
            HTML_TEMPLATE,
            indexed=True,
            message=f"✅ {file.filename} indexed successfully! ({len(chunks)} chunks)",
            chunk_count=total_chunks,
            documents=docs,
            config=Config
        )
    except Exception as e:
        if is_ajax:
            return jsonify({"success": False, "error": str(e)}), 500
        return render_template_string(
            HTML_TEMPLATE,
            error=f"Error indexing document: {str(e)}",
            indexed=get_chunk_count() > 0,
            chunk_count=get_chunk_count(),
            documents=get_uploaded_documents(),
            config=Config
        ), 500

@app.route('/ask', methods=['POST'])
def ask():
    is_ajax = request.headers.get('Accept') == 'application/json' or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json
    
    if request.is_json:
        data = request.get_json() or {}
        question = data.get('question', '').strip()
    else:
        question = request.form.get('question', '').strip()

    if not question:
        if is_ajax:
            return jsonify({"success": False, "error": "Please enter a question"}), 400
        return render_template_string(HTML_TEMPLATE, error="Please enter a question", indexed=get_chunk_count() > 0, chunk_count=get_chunk_count(), documents=get_uploaded_documents(), config=Config)

    chunk_count = get_chunk_count()
    if chunk_count == 0 and len(get_uploaded_documents()) == 0:
        if is_ajax:
            return jsonify({"success": False, "error": "Please upload and index a PDF document first."}), 400
        return render_template_string(HTML_TEMPLATE, error="Please upload and index a PDF document first.", indexed=False, chunk_count=0, documents=[], config=Config)

    try:
        answer = answer_query(question)
        if is_ajax:
            return jsonify({
                "success": True,
                "question": question,
                "answer": answer
            })
        
        return render_template_string(
            HTML_TEMPLATE,
            indexed=True,
            question=question,
            answer=answer,
            chunk_count=chunk_count,
            documents=get_uploaded_documents(),
            config=Config
        )
    except Exception as e:
        if is_ajax:
            return jsonify({"success": False, "error": f"LLM error: {str(e)}"}), 500
        return render_template_string(
            HTML_TEMPLATE,
            error=f"Query error: {str(e)}",
            indexed=True,
            question=question,
            chunk_count=chunk_count,
            documents=get_uploaded_documents(),
            config=Config
        ), 500

@app.route('/delete_document', methods=['POST'])
def delete_document():
    data = request.get_json() if request.is_json else request.form
    filename = data.get('filename', '').strip() if data else ''
    if not filename:
        return jsonify({"success": False, "error": "Filename is required"}), 400

    # Remove only from ChromaDB vector store (local file is preserved on disk)
    remove_documents_by_source(filename)

    total_chunks = get_chunk_count()
    docs = get_uploaded_documents()

    return jsonify({
        "success": True,
        "message": f"Removed '{filename}' from workspace index. Local file remains safe in your folder.",
        "total_chunks": total_chunks,
        "documents": docs
    })

@app.route('/api/status')
def status():
    return jsonify({
        "chunk_count": get_chunk_count(),
        "documents": get_uploaded_documents(),
        "default_model": Config.DEFAULT_MODEL,
        "embedding_model": Config.EMBEDDING_MODEL
    })


if __name__ == '__main__':
    import os
    PORT = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=PORT, debug=False)