import re

entity_unit_map = {
    'width': {'centimetre': 'cm', 'foot': 'ft', 'inch': 'in', 'metre': 'm', 'millimetre': 'mm', 'yard': 'yd'},
    'depth': {'centimetre': 'cm', 'foot': 'ft', 'inch': 'in', 'metre': 'm', 'millimetre': 'mm', 'yard': 'yd'},
    'height': {'centimetre': 'cm', 'foot': 'ft', 'inch': 'in', 'metre': 'm', 'millimetre': 'mm', 'yard': 'yd'},
    'item_weight': {'gram': 'g', 'kilogram': 'kg', 'microgram': 'µg', 'milligram': 'mg', 'ounce': 'oz', 'pound': 'lb', 'ton': 't'},
    'maximum_weight_recommendation': {'gram': 'g', 'kilogram': 'kg', 'microgram': 'µg', 'milligram': 'mg', 'ounce': 'oz', 'pound': 'lb', 'ton': 't'},
    'voltage': {'kilovolt': 'kV', 'millivolt': 'mV', 'volt': 'V'},
    'wattage': {'kilowatt': 'kW', 'watt': 'W'},
    'item_volume': {'centilitre': 'cl', 'cubic foot': 'ft³', 'cubic inch': 'in³', 'cup': 'cup', 'decilitre': 'dl', 
                    'fluid ounce': 'fl oz', 'gallon': 'gal', 'imperial gallon': 'imp gal', 'litre': 'l', 'microlitre': 'µl',
                    'millilitre': 'ml', 'pint': 'pt', 'quart': 'qt'}
}

def regex(text, entity_type):
    # Get the allowed units and their abbreviations for the specified entity type
    allowed_units = entity_unit_map.get(entity_type, {})
    
    # Create a regex pattern that matches both full forms and abbreviations
    unit_patterns = []
    for full_form, abbreviation in allowed_units.items():
        unit_patterns.append(re.escape(full_form))  # Add the full form
        unit_patterns.append(re.escape(abbreviation))  # Add the abbreviation
    
    # Join the unit patterns into a regex pattern (e.g., 'kg|g|gram|grammes')
    unit_pattern = '|'.join(unit_patterns)
    
    # This pattern matches numbers followed by a unit (specific to the entity type)
    pattern = rf'(\d+(?:\.\d+)?)\s*({unit_pattern})'
    
    # Find all matches in the text
    matches = re.findall(pattern, text.lower())
    
    # If matches are found, return the first match as a string (e.g., '34 gram')
    if matches:
        return f"{matches[0][0]} {matches[0][1]}"
    
    return ""

def combine_cols(row):
    # Extract measurement from OCR output using regex
    ocr_text = row['output']
    extracted_measurement = regex(ocr_text)
    
    # Combine relevant columns (e.g., measurement type + OCR text)
    combined_text = f"Type: {row['entity_name']} | OCR: {ocr_text} | Extracted Measurement: {extracted_measurement}"
    
    return combined_text

def preprocess_data(row, tokenizer, max_input_length=128, max_target_length=128):
    inputs = row['model_input']  # Your OCR + regex-preprocessed data
    targets = row['entity_value']  # Extracted measurements
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
