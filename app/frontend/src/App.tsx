import { FormEvent, useEffect, useState } from "react";
import { NotReadyError, sendQuestion } from "./api";
import { LANGUAGES, STRINGS, UiLang } from "./i18n";

interface Message {
  role: "user" | "bot";
  text: string;
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
    setMessages((current) => [...current, { role: "user", text }]);
    setQuestion("");
    setBusy(true);
    let reply: string;
    try {
      reply = (await sendQuestion(text, image)).answer;
      setImage(null);
    } catch (error) {
      reply = error instanceof NotReadyError ? t.notReady : t.error;
    }
    setMessages((current) => [...current, { role: "bot", text: reply }]);
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
          <p
            key={i}
            className={`bubble ${message.role} ${isUrduScript(message.text) ? "urdu" : ""}`}
            dir="auto"
          >
            {message.text}
          </p>
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
