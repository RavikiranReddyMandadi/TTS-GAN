# From https://github.com/keithito/tacotron

import re
from unidecode import unidecode
import inflect
from .cmudict import valid_symbols

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')

# Defines the set of symbols used in text input to the model.
_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

_arpabet = ['@' + s for s in valid_symbols]
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

def _remove_commas(m):
    return m.group(1).replace(',', '')

def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')

def _expand_dollars(m):
    
    match = m.group(1)
    parts = match.split('.')

    if len(parts) > 2:
        return match + ' dollars'

    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0

    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '{} {}, {} {}'.format(dollars, dollar_unit, cents, cent_unit)

    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '{} {}'.format(dollars, dollar_unit)

    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '{} {}'.format(cents, cent_unit)

    else:
        return 'zero dollars'

def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))

def _expand_number(m):
    num = int(m.group(0))

    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)

        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'

        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')

    else:
        return _inflect.number_to_words(num, andword='')

def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text



def clean_text(text):
    # Convert number to text and abbreviation expansion

    # Convert to ascii
    text = unidecode(text)

    # Lower case
    text = text.lower()

    # Expand numbers
    text = normalize_numbers(text)

    # Expand abbreviations
    text = expand_abbreviations(text)

    # Remove white space
    text = re.sub(_whitespace_re, ' ', text)

    return text

def get_arpabet(word, cmu_dict):
  word_arpabet = cmu_dict.lookup(word)
  if word_arpabet is not None:
    return "{" + word_arpabet[0] + "}"
  else:
    return word

def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'

def encode_text(text, cmu_dict=None):

    sequence = []

    space = _symbols_to_sequence(' ')

    # Check for curly braces
    while len(text):
        m = _curly_re.match(text)

        if not m:
            cleaned_text = clean_text(text)

            if cmu_dict is not None:
                cleaned_text = [get_arpabet(w, cmu_dict) for w in cleaned_text.split(" ")] 

                for i in range(len(cleaned_text)):
                    t = cleaned_text[i]

                    if t.startswith("{"):
                        sequence += _arpabet_to_sequence(t[1:-1])
                    else:
                        sequence += _symbols_to_sequence(t)

                    sequence += space

            else:
                sequence += _symbols_to_sequence(cleaned_text)
            
            break

        sequence += _symbols_to_sequence(clean_text(m.group(1)))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    # remove white space
    if cmu_dict is not None:
        sequence = sequence[:-1] if sequence[-1] == space[0] else sequence

    return sequence