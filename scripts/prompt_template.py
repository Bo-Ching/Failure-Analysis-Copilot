DATA_AUGMENTATION_PROMPT = """
Please enhance the following data by rephrasing the content of each field while keeping its original meaning intact.
Use different vocabulary, sentence structures, or more concise expressions. The output must preserve the semantic integrity of the input.

Here is the original data: {texts}

Please output up to 3 enhanced variations in the following format:

Enhanced 1:
description: (Enhanced text)
WHY 1: (Enhanced text)
WHY 2: (Enhanced text)
WHY 3: (Enhanced text)
WHY 4: (Enhanced text)
WHY 5: (Enhanced text)
Corrective Action Verification: (Enhanced text)

Enhanced 2:
description: (Enhanced text)
WHY 1: (Enhanced text)
WHY 2: (Enhanced text)
WHY 3: (Enhanced text)
WHY 4: (Enhanced text)
WHY 5: (Enhanced text)
Corrective Action Verification: (Enhanced text)

Enhanced 3:
description: (Enhanced text)
WHY 1: (Enhanced text)
WHY 2: (Enhanced text)
WHY 3: (Enhanced text)
WHY 4: (Enhanced text)
WHY 5: (Enhanced text)
Corrective Action Verification: (Enhanced text)

Instructions:
1. Ensure that the meaning of each field remains consistent with the original.
2. Use alternative vocabulary, grammar structures, or expressions to rephrase the text.
3. The output should be clear and easy to understand.
4. If no enhancement is needed for a field, just return None.
5. Maintain the language of the original field (e.g., Chinese text stays in Chinese, English text stays in English).
"""
