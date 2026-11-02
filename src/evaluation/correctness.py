import re
import unicodedata


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def is_correct(parsed, truth, choices):
    if not parsed.valid:
        return False
    return parsed.answer == truth if choices else normalize(parsed.answer) == normalize(truth)
