import unidecode
import re

def firm_name_clean(firm_name, lower=True, remove_punc=True, remove_legal=True, remove_parentheses=True):
    # make string
    firm_name = str(firm_name)
    firm_name = unidecode.unidecode(firm_name)
    # lowercase
    if lower:
        firm_name = firm_name.lower()
    # remove punctuation
    if remove_punc:
        firm_name = firm_name.translate(str.maketrans('', '', '!"#$%\\\'*+,./:;<=>?@^_`{|}~'))
    # remove legal identifiers
    if remove_legal:
        legal_identifiers = ["co", "inc", "ag", "ltd", "lp", "llc", "pllc", "llp", "plc", "ltdplc", "corp",
                             "corporation", "ab", "cos", "cia", "sa", "company", "companies", "consolidated",
                             "stores", "limited", "srl", "kk", "gmbh", "pty", "group", "yk", "bhd",
                             "limitada", "holdings", "kg", "bv", "pte", "sas", "ilp", "nl", "genossenschaft",
                             "gesellschaft", "aktiengesellschaft", "ltda", "nv", "oao", "holding", "se",
                             "oy", "plcnv", "the", "neft", "& co", "&co"]
        pattern = '|'.join(legal_identifiers)
        pattern = '\\b(' + pattern + ')\\b'  # match only word boundaries
        firm_name = re.sub(pattern, '', firm_name)
    # remove parentheses and anything in them: Bayerische Motoren Werke (BMW) -> Bayerische Motoren Werke
    if remove_parentheses:
        firm_name = re.sub(r'\([^()]*\)', '', firm_name)

    # make hyphens consistent
    firm_name = firm_name.replace(' - ', '-')

    # remove ampersand symbol
    firm_name = firm_name.replace('&amp;', '&')
    firm_name = firm_name.replace('&amp', '&')

    # strip
    firm_name = firm_name.strip()

    return firm_name

def extract_firm_name_and_spans(text, names, clean_names=True):
    if clean_names:
        cleaned_names = [firm_name_clean(name) for name in names]
        names += cleaned_names
    pattern = r'|'.join(re.escape(word.strip()) for word in names)

    res = re.finditer(pattern, text, flags=re.IGNORECASE)

    return [(match.group(), match.span()) for match in res]

def clean_unique_entities(ents):
    seen_ents = []
    res = []
    for ent in ents:
        cleaned_ent = firm_name_clean(ent[0])
        if cleaned_ent not in seen_ents:
            res.append(ent + (cleaned_ent,))
            seen_ents.append(cleaned_ent)


    return res
