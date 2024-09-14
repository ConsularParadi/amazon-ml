import re
import pandas as pd

unit_standardization = {
    'cm': 'centimetre',
    'mm': 'millimetre',
    'm': 'metre',
    'ft': 'foot',
    'in': 'inch',
    'yd': 'yard',
    'g': 'gram',
    'kg': 'kilogram',
    'mg': 'milligram',
    'µg': 'microgram',
    'oz': 'ounce',
    'lb': 'pound',
    't': 'ton',
    'ml': 'millilitre',
    'l': 'litre',
    'kv': 'kilovolt',
    'mv': 'millivolt',
    'v': 'volt',
    'kw': 'kilowatt',
    'w': 'watt'
}

# entity to unit mapping
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}


class Postprocess:
    def __init__(self):
        pass

    def __standardize_units(self, text):
        text = text.lower()  # Normalize case
        for short_form, full_form in unit_standardization.items():
            text = re.sub(rf'\\b{short_form}\\b', full_form, text)
        return text

    def __preprocess_text(self, text):
        text = text.replace('\\n', ' ')
        text = re.sub(r'(\\d)([a-zA-Z])', r'\\1 \\2', text)
        text = re.sub(r'([a-zA-Z])(\\d)', r'\\1 \\2', text)
        text = re.sub(r'\\s+', ' ', text).strip()
        return text

    def __extract_values_and_units(self, text):
        pattern = r'\\b(\\d+(\\.\\d+)?)\\s*([a-zA-Z]+)?\\b'
        matches = re.findall(pattern, text)
        values = [match[0] for match in matches]
        units = [match[2] for match in matches if match[2]]
        return values, units

    def __assign_default_units(self, row):
        if not row['units']:  # If the units list is empty
            default_unit = next(iter(entity_unit_map.get(row['entity_name'], [])), '')
            row['units'] = [default_unit] if default_unit else []
        return row

    def process_units(self, df: pd.DataFrame):
        df['output'] = df['output'].apply(lambda x: self.__standardize_units(x))
        df['processed_output'] = df['output'].apply(lambda x: self.__preprocess_text(x))
        df['numerical_values'], df['units'] = zip(*df['processed_output'].apply(lambda x: self.__extract_values_and_units(x)))
        df = df.apply(self.assign_default_units, axis=1)
        return df  