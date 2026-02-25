/**
 * OmniDraft API Client
 * Handles all communication with the FastAPI backend.
 */

import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_URL,
  timeout: 600000, // 10 min timeout for large multi-page PDFs
});

// ── Upload ────────────────────────────────────────────────────────────────────

export async function uploadFile(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 600000, // 10 min for upload + processing
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded * 100) / e.total));
      }
    },
  });
  return response.data;
}

// ── Notes ─────────────────────────────────────────────────────────────────────

export async function fetchNotes() {
  const response = await api.get('/notes');
  return response.data;
}

export async function fetchNote(noteId) {
  const response = await api.get(`/notes/${noteId}`);
  return response.data;
}

export async function deleteNote(noteId) {
  const response = await api.delete(`/notes/${noteId}`);
  return response.data;
}

export async function updateNoteText(noteId, updatedText) {
  const response = await api.patch(`/notes/${noteId}`, null, {
    params: { updated_text: updatedText },
  });
  return response.data;
}

// ── Export ─────────────────────────────────────────────────────────────────────

export function getDownloadUrl(noteId, format = 'pdf') {
  return `${API_URL}/download/${noteId}?format=${format}`;
}

export async function downloadNote(noteId, format = 'pdf') {
  const response = await api.get(`/download/${noteId}`, {
    params: { format },
    responseType: 'blob',
    timeout: 30000,
  });

  // Trigger browser download
  const contentDisposition = response.headers['content-disposition'];
  let filename = `note.${format}`;
  if (contentDisposition) {
    const match = contentDisposition.match(/filename="?(.+?)"?$/);
    if (match) filename = match[1];
  }

  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

// ── RAG Chat ──────────────────────────────────────────────────────────────────

export async function chatWithNotes(query, noteId = null) {
  const response = await api.post('/chat', {
    query,
    note_id: noteId,
  });
  return response.data;
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function checkHealth() {
  try {
    const response = await api.get('/health', { timeout: 5000 });
    return response.data;
  } catch {
    return { status: 'error', version: 'unknown', llm_provider: 'unknown' };
  }
}

export default api;
