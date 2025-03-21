from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langdetect import detect
from transformers import pipeline

### 1. Translation (as needed)
# Toucan
toucan_lang_names={
    "aar": "Afar",
    "ach": "Acholi",
    "afr": "Afrikaans",
    "aka": "Akan",
    "amh": "Amharic",
    "bam": "Bambara",
    "bas": "Basaa",
    "bem": "Bemba",
    "btg": "Bete Gagnoa",
    "eng": "English",
    "ewe": "Ewe",
    "fon": "Fon",
    "fra": "French",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kbp": "Kabiye",
    "lgg": "Lugbara",
    "lug": "Luganda",
    "mlg": "Malagasy",
    "nyn": "Nyakore",
    "orm": "Oromo",
    "som": "Somali",
    "sot": "Sesotho",
    "swa": "Swahili",
    "tir": "Tigrinya",
    "yor": "Yoruba",
    "teo": "Ateso",
    "gez": "Geez",
    "wal": "Wolaytta",
    "fan": "Fang",
    "kau": "Kanuri",
    "kin": "Kinyawanda",
    "kon": "Kongo",
    "lin": "Lingala",
    "nya": "Chichewa",
    "pcm": "Nigerian Pidgin",
    "ssw": "Siswati",
    "tsn": "Setswana",
    "tso": "Tsonga",
    "twi": "Twi",
    "wol": "Wolof",
    "xho": "Xhosa",
    "zul": "Zulu",
    "nnb": "Nande",
    "swc": "Swahili Congo",
    "ara": "Arabic"
}
toucan_langs = [lang for lang in toucan_lang_names.keys()]

# granite-3.0
granite_langs = [
    "en",  # English
    "de",  # German
    "es",  # Spanish
    "fr",  # French
    "ja",  # Japanese
    "pt",  # Portuguese
    "ar",  # Arabic
    "cs",  # Czech
    "it",  # Italian
    "ko",  # Korean
    "nl",  # Dutch
    "zh"   # Chinese
]

# FLORES 200 language - this is a map as FLORES uses a distinct language code
flores_langs_map = langdetect_to_flores = {
    "ace": "ace_Latn",
    "acm": "acm_Arab",
    "acq": "acq_Arab",
    "aeb": "aeb_Arab",
    "af": "afr_Latn",
    "ajp": "ajp_Arab",
    "ak": "aka_Latn",
    "am": "amh_Ethi",
    "apc": "apc_Arab",
    "ar": "arb_Arab",
    "ars": "ars_Arab",
    "ary": "ary_Arab",
    "arz": "arz_Arab",
    "as": "asm_Beng",
    "ast": "ast_Latn",
    "awa": "awa_Deva",
    "ayr": "ayr_Latn",
    "azb": "azb_Arab",
    "az": "azj_Latn",
    "ba": "bak_Cyrl",
    "bm": "bam_Latn",
    "ban": "ban_Latn",
    "be": "bel_Cyrl",
    "bem": "bem_Latn",
    "bn": "ben_Beng",
    "bho": "bho_Deva",
    "bjn": "bjn_Latn",
    "bo": "bod_Tibt",
    "bs": "bos_Latn",
    "bug": "bug_Latn",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
    "ceb": "ceb_Latn",
    "cs": "ces_Latn",
    "cjk": "cjk_Latn",
    "ckb": "ckb_Arab",
    "crh": "crh_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "dik": "dik_Latn",
    "dyu": "dyu_Latn",
    "dz": "dzo_Tibt",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "eo": "epo_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "ee": "ewe_Latn",
    "fo": "fao_Latn",
    "fj": "fij_Latn",
    "fi": "fin_Latn",
    "fon": "fon_Latn",
    "fr": "fra_Latn",
    "fur": "fur_Latn",
    "ff": "fuv_Latn",
    "gd": "gla_Latn",
    "ga": "gle_Latn",
    "gl": "glg_Latn",
    "gn": "grn_Latn",
    "gu": "guj_Gujr",
    "ht": "hat_Latn",
    "ha": "hau_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hne": "hne_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "ig": "ibo_Latn",
    "ilo": "ilo_Latn",
    "id": "ind_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "jv": "jav_Latn",
    "ja": "jpn_Jpan",
    "kab": "kab_Latn",
    "kac": "kac_Latn",
    "kam": "kam_Latn",
    "kn": "kan_Knda",
    "ks": "kas_Arab",
    "ka": "kat_Geor",
    "kr": "kau_Latn",
    "kk": "kaz_Cyrl",
    "kbp": "kbp_Latn",
    "kea": "kea_Latn",
    "km": "khm_Khmr",
    "ki": "kik_Latn",
    "rw": "kin_Latn",
    "ky": "kir_Cyrl",
    "kmb": "kmb_Latn",
    "ku": "kmr_Latn",
    "kg": "kon_Latn",
    "ko": "kor_Hang",
    "lo": "lao_Laoo",
    "lij": "lij_Latn",
    "li": "lim_Latn",
    "ln": "lin_Latn",
    "lt": "lit_Latn",
    "lmo": "lmo_Latn",
    "ltg": "ltg_Latn",
    "lb": "ltz_Latn",
    "lua": "lua_Latn",
    "lg": "lug_Latn",
    "luo": "luo_Latn",
    "lus": "lus_Latn",
    "lv": "lvs_Latn",
    "mag": "mag_Deva",
    "mai": "mai_Deva",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "min": "min_Latn",
    "mk": "mkd_Cyrl",
    "plt": "plt_Latn",
    "mt": "mlt_Latn",
    "mni": "mni_Beng",
    "mn": "mon_Cyrl",
    "mos": "mos_Latn",
    "mi": "mri_Latn",
    "my": "mya_Mymr",
    "nl": "nld_Latn",
    "nn": "nno_Latn",
    "nb": "nob_Latn",
    "ne": "npi_Deva",
    "nso": "nso_Latn",
    "nus": "nus_Latn",
    "ny": "nya_Latn",
    "oc": "oci_Latn",
    "om": "orm_Latn",
    "or": "ory_Orya",
    "pag": "pag_Latn",
    "pa": "pan_Guru",
    "pap": "pap_Latn",
    "fa": "pes_Arab",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "prs": "prs_Arab",
    "ps": "pus_Arab",
    "quy": "quy_Latn",
    "ro": "ron_Latn",
    "rn": "run_Latn",
    "ru": "rus_Cyrl",
    "sg": "sag_Latn",
    "sa": "san_Deva",
    "sat": "sat_Olck",
    "scn": "scn_Latn",
    "shn": "shn_Mymr",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sm": "smo_Latn",
    "sn": "sna_Latn",
    "sd": "snd_Arab",
    "so": "som_Latn",
    "st": "sot_Latn",
    "es": "spa_Latn",
    "srd": "srd_Latn",
    "sr": "srp_Cyrl",
    "ss": "ssw_Latn",
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "szl": "szl_Latn",
    "ta": "tam_Taml",
    "tt": "tat_Cyrl",
    "te": "tel_Telu",
    "tg": "tgk_Cyrl",
    "tl": "tgl_Latn",
    "th": "tha_Thai",
    "ti": "tir_Ethi",
    "tpi": "tpi_Latn",
    "tn": "tsn_Latn",
    "ts": "tso_Latn",
    "tk": "tuk_Latn",
    "tum": "tum_Latn",
    "tr": "tur_Latn",
    "tw": "twi_Latn",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "war": "war_Latn",
    "wo": "wol_Latn",
    "xh": "xho_Latn",
    "yo": "yor_Latn",
    "yue": "yue_Hant",
    "zh": "zho_Hans"
}

def get_translation_model(language):
    """Selects the appropriate translation model based on the language."""
    model_name = None
    src_lang = None
    tgt_lang = "eng_Latn"  # Target language is always English

    if language in toucan_langs:
        model_name = "UBC-NLP/toucan-base"
        src_lang = language
    elif language in flores_langs_map.keys():
        model_name = "facebook/nllb-200-distilled-600M"
        src_lang = flores_langs_map[language]
    else:
        raise ValueError(f"Language not supported: {language}")

    # Load the translation pipeline
    translator = pipeline("translation", model=model_name)
    return translator, src_lang, tgt_lang

def translate_text(text):
    """Translates non-English text to English using the selected model."""
    detected_lang = detect(text)
    
    if detected_lang == "en":
        return text  # No translation needed
    
    translator, src_lang, tgt_lang = get_translation_model(detected_lang)

    # Ensure proper function call for the pipeline
    translated_text = translator(text, src_lang=src_lang, tgt_lang=tgt_lang)[0]["translation_text"]
    return translated_text

# Example Usage
if __name__ == "__main__":
    text = "Hola, ¿cómo estás?"
    translated_text = translate_text(text)
    print(f"Original: {text}\nTranslated: {translated_text}")
