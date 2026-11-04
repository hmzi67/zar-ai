const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface AnswerResponse {
  answer: string;
  lang: "ur" | "ur-Latn" | "en";
  status: "answered" | "clarify" | "refused" | "blocked";
  sources: string[];
}

export class NotReadyError extends Error {}

async function parse(response: Response): Promise<AnswerResponse> {
  if (response.status === 501) throw new NotReadyError();
  if (!response.ok) throw new Error(`API error ${response.status}`);
  return response.json();
}

export async function sendQuestion(question: string, image: File | null): Promise<AnswerResponse> {
  if (image) {
    const form = new FormData();
    form.append("question", question);
    form.append("image", image);
    return parse(await fetch(`${API_URL}/ask`, { method: "POST", body: form }));
  }
  return parse(
    await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  );
}
