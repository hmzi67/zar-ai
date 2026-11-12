export type UiLang = "ur" | "ur-Latn" | "en";

export const LANGUAGES: { code: UiLang; label: string }[] = [
  { code: "ur", label: "اردو" },
  { code: "ur-Latn", label: "Roman Urdu" },
  { code: "en", label: "English" },
];

export interface Strings {
  placeholder: string;
  send: string;
  photo: string;
  remove: string;
  language: string;
  notReady: string;
  error: string;
  sending: string;
  badImage: string;
  modelSaw: string;
  lowConfidence: string;
  diseases: Record<string, string>;
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
    badImage: "یہ تصویر کھل نہیں سکی، کوئی اور تصویر لگائیں۔",
    modelSaw: "تصویر کے ماڈل کے امکانات",
    lowConfidence: "ماڈل کو یقین نہیں",
    diseases: {
      healthy: "صحت مند",
      yellow_rust: "پیلی کنگی",
      brown_rust: "بھوری کنگی",
      black_rust: "کالی کنگی",
      septoria: "سیپٹوریا",
    },
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
    badImage: "Yeh tasveer khul nahi saki, koi aur tasveer lagayein.",
    modelSaw: "Tasveer ke model ke imkanaat",
    lowConfidence: "Model ko yaqeen nahi",
    diseases: {
      healthy: "Sehatmand",
      yellow_rust: "Peeli kungi",
      brown_rust: "Bhoori kungi",
      black_rust: "Kaali kungi",
      septoria: "Septoria",
    },
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
    badImage: "That photo could not be read, please attach another.",
    modelSaw: "Image model's top guesses",
    lowConfidence: "the model isn't confident",
    diseases: {
      healthy: "Healthy",
      yellow_rust: "Yellow (stripe) rust",
      brown_rust: "Brown (leaf) rust",
      black_rust: "Black (stem) rust",
      septoria: "Septoria",
    },
  },
};
