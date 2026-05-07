import axios from "axios";

const defaultBaseUrl = `${window.location.protocol}//${window.location.hostname}:8001`;
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || defaultBaseUrl;

const api = axios.create({
  baseURL: API_BASE_URL
});

export async function enhanceAudio(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/enhance", form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
}
