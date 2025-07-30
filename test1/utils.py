import pandas as pd

# 여러 컬럼 문자열 수동 파싱
# 사용법 : list_cols =['파', '싱', '할', '컬', '럼']
#          df = parse_string_list_column(df, list_cols)
def parse_string_list_column(df, columns):
    def parse(text):
        if pd.isna(text):
            return []
        text = text.strip().strip('[]')
        items = text.split(',')
        return [i.strip().strip("'\"") for i in items if i.strip()]
    
    for col in columns:
        new_col = f"{col}_parsed"
        df[new_col] = df[col].apply(parse)
    
    return df

# 수동 매핑 적용
# mapping 딕셔너리 선언
# 적용 ) df['매핑 적용할 변수'] = normalize_list_column(df, '매핑 적용할 변수', mapping)
def normalize_list_column(df, column, mapping):
    """
    리스트형 문자열 컬럼에서 매핑에 따라 항목 정리

    예: ['풀잎향', '풀향'] → ['풀잎 향']
    """
    def normalize(item_list):
        return [mapping.get(i.strip(), i.strip()) for i in item_list if i.strip()]

    return df[column].apply(normalize)


# 벡터화 모듈
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def vectorize_multilabel_column(df, col_name, prefix=None):
    """
    multilabel column을 MultiLabelBinarizer로 벡터화하고 DataFrame 반환
    
    Parameters:
    - df: 원본 데이터프레임
    - col_name: 벡터화할 리스트 컬럼명 (예: 'Aroma_parsed')
    - prefix: 생성될 컬럼 접두어 (예: 'Aroma_'). 생략 시 col_name 기준 자동 생성

    Returns:
    - vector_df: 벡터화된 DataFrame
    - mlb: MultiLabelBinarizer 객체 (나중에 역변환 등에 사용 가능)
    """
    mlb = MultiLabelBinarizer()
    vector = mlb.fit_transform(df[col_name])

    if prefix is None:
        prefix = col_name.replace('_parsed', '')

    columns = [f"{prefix}_{cls}" for cls in mlb.classes_]
    vector_df = pd.DataFrame(vector, columns=columns, index=df.index)

    return vector_df, mlb
