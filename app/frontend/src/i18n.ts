export type UiLang = "ur" | "ur-Latn" | "en";

export const LANGUAGES: { code: UiLang; label: string }[] = [
  { code: "ur", label: "اردو" },
  { code: "ur-Latn", label: "Roman Urdu" },
  { code: "en", label: "English" },
];

interface Strings {
  placeholder: string;
  send: string;
  photo: string;
  remove: string;
  language: string;
  notReady: string;
  error: string;
  sending: string;
}

export const STRINGS: Record<UiLang, Strings> = {
  ur: {
    placeholder: "اپنی فصل کے بارے میں سوال لکھیں",
    send: "بھیجیں",
    photo: "فصل کی تصویر لگائیں",
    remove: "تصویر ہٹائیں",
    language: "زبان",
    notReady: "نظام ابھی تیار نہیں ہے۔",
    error: "کچھ غلط ہو گیا، دوبارہ کوشش کریں۔",
    sending: "جواب آ رہا ہے",
  },
  "ur-Latn": {
    placeholder: "Apni fasal ke baare mein sawal likhein",
    send: "Bhejein",
    photo: "Fasal ki tasveer lagayein",
    remove: "Tasveer hatayein",
    language: "Zaban",
    notReady: "Nizam abhi tayyar nahi hai.",
    error: "Kuch ghalat ho gaya, dobara koshish karein.",
    sending: "Jawab aa raha hai",
  },
  en: {
    placeholder: "Ask a question about your crop",
    send: "Send",
    photo: "Attach a crop photo",
    remove: "Remove photo",
    language: "Language",
    notReady: "The system is not ready yet.",
    error: "Something went wrong, please try again.",
    sending: "Getting an answer",
  },
};
