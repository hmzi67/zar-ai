const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface Prediction {
  label: string;
  prob: number;
}

export interface Vision {
  top: Prediction[];
  abstain: boolean;
  // confident >= 0.90, likely >= 0.60, unsure below; not_plant: turned away, top is empty.
  band: "confident" | "likely" | "unsure" | "not_plant";
  plant_prob: number | null;
}

export interface AnswerResponse {
  answer: string;
  lang: "ur" | "ur-Latn" | "en";
  // diagnosis: the image model's result alone, while the knowledge base or generator isn't ready.
  status: "answered" | "clarify" | "refused" | "blocked" | "diagnosis" | "not_plant";
  sources: string[];
  vision: Vision | null;
}

export class NotReadyError extends Error {}
export class BadImageError extends Error {}

async function parse(response: Response): Promise<AnswerResponse> {
  if (response.status === 501) throw new NotReadyError();
  if (response.status === 400) throw new BadImageError();
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
