import { FormEvent, useEffect, useState } from "react";
import { BadImageError, NotReadyError, sendQuestion, Vision } from "./api";
import { LANGUAGES, STRINGS, Strings, UiLang } from "./i18n";

interface Message {
  role: "user" | "bot";
  text: string;
  imageUrl?: string;
  vision?: Vision | null;
}

function Predictions({ vision, t }: { vision: Vision; t: Strings }) {
  return (
    <div className="predictions">
      <span className="predictions-title">
        {t.modelSaw}
        {vision.band === "unsure" && ` (${t.lowConfidence})`}
        {vision.band === "likely" && ` (${t.likely})`}
      </span>
      {vision.top.map(({ label, prob }) => (
        <div key={label} className="prediction">
          <span className="prediction-label">{t.diseases[label] ?? label}</span>
          <span className="prediction-bar" aria-hidden="true">
            <span style={{ width: `${Math.round(prob * 100)}%` }} />
          </span>
          <span className="prediction-prob">{(prob * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

// Each message is displayed by the script it was written in, not by the UI language.
const isUrduScript = (text: string) => /[؀-ۿ]/.test(text);

export default function App() {
  const [lang, setLang] = useState<UiLang>("ur");
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const t = STRINGS[lang];

  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "ur" ? "rtl" : "ltr";
  }, [lang]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;
    const imageUrl = image ? URL.createObjectURL(image) : undefined;
    setMessages((current) => [...current, { role: "user", text, imageUrl }]);
    setQuestion("");
    setBusy(true);
    let reply: Message;
    try {
      const response = await sendQuestion(text, image);
      reply = { role: "bot", text: response.answer, vision: response.vision };
      setImage(null);
    } catch (error) {
      const failure =
        error instanceof NotReadyError
          ? t.notReady
          : error instanceof BadImageError
            ? t.badImage
            : t.error;
      reply = { role: "bot", text: failure };
    }
    setMessages((current) => [...current, reply]);
    setBusy(false);
  }

  return (
    <main className="app">
      <header className="header">
        <h1>ZarAI</h1>
        <label>
          {t.language}{" "}
          <select value={lang} onChange={(e) => setLang(e.target.value as UiLang)}>
            {LANGUAGES.map(({ code, label }) => (
              <option key={code} value={code}>
                {label}
              </option>
            ))}
          </select>
        </label>
      </header>

      <section className="chat" aria-live="polite">
        {messages.map((message, i) => (
          <div key={i} className={`bubble ${message.role}`}>
            {message.imageUrl && <img className="photo" src={message.imageUrl} alt="" />}
            <p className={isUrduScript(message.text) ? "urdu" : ""} dir="auto">
              {message.text}
            </p>
            {message.vision && message.vision.top.length > 0 && (
              <Predictions vision={message.vision} t={t} />
            )}
          </div>
        ))}
        {busy && <p className="status">{t.sending}…</p>}
      </section>

      <form className="composer" onSubmit={submit}>
        <textarea
          aria-label={t.placeholder}
          placeholder={t.placeholder}
          value={question}
          maxLength={2000}
          rows={2}
          dir="auto"
          onChange={(e) => setQuestion(e.target.value)}
        />
        <div className="actions">
          <label className="upload">
            {t.photo}
            <input
              type="file"
              accept="image/*"
              capture="environment"
              onChange={(e) => setImage(e.target.files?.[0] ?? null)}
            />
          </label>
          {image && (
            <button type="button" onClick={() => setImage(null)}>
              {t.remove}: {image.name}
            </button>
          )}
          <button type="submit" disabled={busy || !question.trim()}>
            {t.send}
          </button>
        </div>
      </form>
    </main>
  );
}
