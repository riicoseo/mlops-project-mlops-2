import ast
import pandas as pd
from typing import List

def normalize_list_columns(df: pd.DataFrame) -> List[str]:

    list_cols = []

    for col in df.columns:
        sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None

        is_list_col = False
        if isinstance(sample_val, list):
            is_list_col = True
        elif isinstance(sample_val, tuple) and len(sample_val) == 1 and isinstance(sample_val[0], list):
            # ex.  ([청량함, 견과류 풍미, 스파이시, 쓴 맛])
            is_list_col = True
        elif isinstance(sample_val, str) and sample_val.startswith("[") and sample_val.endswith("]"):
            # ex.  ['쓴 맛', '상쾌함']
            try:
                cleaned_val = sample_val.replace("\"\"", "")
                parsed_val = ast.literal_eval(cleaned_val)
                if isinstance(parsed_val, list):
                    is_list_col = True
            except (ValueError, SyntaxError):
                pass

        if is_list_col:
            list_cols.append(col)

            def convert(val):
                if pd.isna(val):
                    return val
          
                if isinstance(val, list):
                    return val

                if isinstance(val, tuple) and len(val) == 1 and isinstance(val[0], list):
                    return val[0]
                
                if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                    try:
                        parsed = ast.literal_eval(val)
                        if isinstance(parsed, list):
                            return parsed
                    except (ValueError, SyntaxError):
                        return None
                return None  # 그 외는 None 처리

            df[col] = df[col].apply(convert)

    return list_cols
