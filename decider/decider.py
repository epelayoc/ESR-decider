import os
import re
import pandas as pd
from pprint import pprint
from typesafe_sdk import TypeSafeClient
from transformers import AutoModel
import time
import json
from tqdm.auto import tqdm
import torch


class SystemOneClient:
    """Cliente unificado para interactuar con modelos TYPESAFE y LIQUIDAI."""

    def __init__(self, model: str = "LIQUID-AI", model_id: str = "LiquidAI/d1-2B", api_key=None):
        """
        Inicializa el cliente según el modelo especificado.

        - model: 'TYPESAFE' o 'LIQUID-AI'
        - model_id: Identificador del modelo para Hugging Face (solo aplica a LIQUIDAI)
        """
        self.model_type = model.upper()
        self.api_key = api_key

        if self.model_type == "TYPESAFE":
            if self.api_key:
                os.environ["TYPESAFE_API_KEY"] = self.api_key
            self.client = TypeSafeClient()

        elif self.model_type == "LIQUID-AI":
            self.client = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,   
                device_map="auto",
            )
            
        else:
            raise ValueError( f"Modelo no válido '{model}'. Selecciona 'TYPESAFE' o 'LIQUID-AI'.")

    def _round_nested(self, val, decimals=3):
        if isinstance(val, dict):
            return {k: self._round_nested(v, decimals) for k, v in val.items()}
        if isinstance(val, float):
            return round(val, decimals)
        return val

    def system_one(self, state: str, questions: dict, **kwargs):
        """
        Ejecuta la función system_one del modelo subyacente.
        - state: Texto de entrada / contexto a analizar.
        - questions: Diccionario con la estructura de preguntas/criterios.
        """
        output = self.client.system_one(state=state, questions=questions, **kwargs)

        if hasattr(output, "model_dump"):
            output = output.model_dump()
        elif hasattr(output, "__dict__"):
            output = vars(output)
        return output

    def eval(self, txt, questions, show_legend=False, condition=None, prob_list=False):
        output = self.system_one(txt, questions)
        answers = output["answers"]
        tokens = output["usage"]["input_tokens"]
        d = {}
        for k in answers.keys():
            a = answers[k]  # answer
            type = a['type']  # type
            d[k] = a[type]  # answer_value
            if type in ['choice', 'score']:  # Add labels confidences & probs

                if prob_list:
                    question = questions[k]
                    criteria = question.get('criteria')
                    probs = a['probabilities']
                    if type == 'score':
                        ordered_probs = [round(probs[k], 2) for k in sorted(probs, key=int)]
                    else: # Choice
                        ordered_probs = [round(probs[k], 2) for k in list(criteria.keys())]

                    # map(str, ...) asegura que convierta enteros o strings correctamente
                    keys_formatted = f"[{', '.join(map(str, probs.keys()))}]"
                    d[f"{k}-p: \n {keys_formatted}"] = ordered_probs


                else:
                    d[k + '-p'] = a['probabilities']
                if show_legend:
                    if type == 'score':
                        # (1) 'score': Leyenda del valor más cercano en a['legend']
                        legend = a.get('legend', {})
                        if legend:
                            score_idx = int(round(float(d[k])))
                            d[k + '-l'] = legend.get(score_idx) or legend.get(str(score_idx))
                    elif type == 'choice':
                        choice_val = d[k] # selected option
                        question = questions[k]
                        if isinstance(question, dict):
                            criteria = question.get('criteria')
                            d[k + '-l'] = criteria.get(choice_val) # takes label
                #d[k + '-d'] = a['probabilities']
                d[k + '-c'] = a['confidence']

        return (d, tokens)


    def split(self, text_block: str, split_words = False) -> list[str]:
        """
        Divide el bloque de texto en párrafos, separa párrafos con conectores de ruptura
        y fusiona párrafos breves que indican debilidades con el párrafo anterior.
        """

        # 1. División inicial en párrafos
        paragraphs = [ p.strip() for p in re.split(r'(?<=\.)\s*\n', text_block) if p.strip() ]

        # 2. Dividir párrafos Q con conectores y >= 2 frases
        if split_words:
            conectores_ruptura = ['however', 'but', 'nevertheless', 'on the other hand', 'nonetheless']
            palabras_clave_debilidad = {'shortcoming', 'shortcomings', 'weakness', 'weaknesses'}

            patron_conectores = r'\b(' + '|'.join(re.escape(c) for c in conectores_ruptura) + r')\b'
            parrafos_divididos = []
            for q in paragraphs:
                match = re.search(patron_conectores, q, re.IGNORECASE)
                if '.' in q and match and match.start() > 0:
                    idx = match.start()
                    q1 = q[:idx].strip()
                    q2 = q[idx:].strip()

                    if q1.endswith(',') or q1.endswith(';'):
                        q1 = q1[:-1].strip() + '.'
                    elif not q1.endswith('.'):
                        q1 = q1 + '.'

                    if q2:
                        q2 = q2[0].upper() + q2[1:]

                    parrafos_divididos.extend([q1, q2])
                else:
                    parrafos_divididos.append(q)

            # 3. Fusionar párrafos cortos P (3 a 5 palabras con palabra clave)
            paragraphs = []
            for p in parrafos_divididos:
                palabras = re.findall(r'\b[\w-]+\b', p)
                tiene_kw = any(w.lower() in palabras_clave_debilidad for w in palabras)

                if 3 <= len(palabras) <= 5 and tiene_kw:
                    if paragraphs:
                        paragraphs[-1] += " " + p
                    else:
                        paragraphs.append(p)
                else:
                    paragraphs.append(p)
        return paragraphs


    def classify( self, text_block, q1, q2, column,
                  eval_cond=None, show_legend=True, split_words=True, index=None ):

        paragraphs = self.split(text_block, split_words)
        rows = []

        for i, p in enumerate(paragraphs):
            start_time = time.perf_counter()  # Inicio de la medición

            # Create a fresh shallow copy of the index dictionary for each row
            d = index.copy() if index else {}
            d['paragraph_num'] = i
            d['txt'] = p

            # 1st eval
            ans1, tokens1 = self.eval(p, q1, prob_list=True)

            # 2nd eval
            isEVAL2 = True if eval_cond is None else ans1[eval_cond[0]] > eval_cond[1]
            if isEVAL2:
                q = q2[ans1[column]]
                ans2, tokens2 = self.eval(p, q,  show_legend=True)
                ans1.update(ans2)
                ans1['tokens'] = tokens1 + tokens2
            else:
                ans1['tokens'] = tokens1

            d.update(ans1)
            d["time"] = time.perf_counter() - start_time
            rows.append(d)
        return rows

    def process_excel1( self, txt_column: str, input_file: str, num_rows: int = None,
                      output_file: str = "resultado_procesado.xlsx", json_file="datos.json", base_path: str = ".",index_cols=[] ):

        """Lee un archivo Excel desde Google Colab, procesa una columna específica
        usando un cliente y guarda/devuelve el resultado.
        """

        # 1. Rutas directas a la carpeta por defecto de Google Colab (/content/)
        #base_path = "/content/"
        #input_path = os.path.join(base_path, input_file)
        #output_path = os.path.join(base_path, self.model_type+"_"+output_file)

        input_path = (
                os.path.join(base_path, input_file)
                if not os.path.isabs(input_file)
                else input_file
            )
        
        output_path = (
            os.path.join(base_path, self.model_type + "_" + output_file)
            if not os.path.isabs(output_file)
            else output_file
        )

        # 2. Cargar el archivo Excel de entrada y el JSON
        with open(json_file, "r", encoding="utf-8") as jsonfile:
            questions = json.load(jsonfile)
        df_in = pd.read_excel(input_path, nrows=num_rows if num_rows else None)

        # 3. Recorrer las filas del DataFrame con barra de progreso
        df_list = []
        for index, r in tqdm( df_in.iterrows(), total=len(df_in), desc="Procesando comentarios" ):
            idx = {}
            for i in index_cols:
                idx[i] = r[i]
            idx['Block'] = txt_column
            comment = r[txt_column]
            output = self.classify( comment, questions["q1"], questions["q2"], "topic_area",
                                        eval_cond=["negativity", 1.5],index=idx, )
            df_list += output

        # 4. Concatenar todos los DataFrames de la lista en uno solo
        df_out = pd.DataFrame(df_list)

        # 5. Guardar el DataFrame final en un archivo Excel en Google Colab
        df_out.to_excel(output_path, index=False)
        print(f"Archivo guardado con éxito en: {output_path}")

        return df_out
                          
def process_excel(
    self,
    txt_column: str,
    input_file: str,
    num_rows: int = None,
    output_file: str = "resultado_procesado.xlsx",
    json_file: str = "datos.json",
    base_path: str = ".",
    index_cols: list = []
):
    """Lee un archivo Excel, procesa una columna especifica usando un cliente y guarda/devuelve el resultado."""

    # 1. Resolver rutas de entrada y salida
    input_path = input_file if os.path.isabs(input_file) else os.path.join(base_path, input_file)
    json_path = json_file if os.path.isabs(json_file) else os.path.join(base_path, json_file)
    
    # La salida siempre se guarda en el directorio actual o base_path especificado
    output_filename = self.model_type + "_" + output_file if not os.path.isabs(output_file) else output_file
    output_path = output_filename if os.path.isabs(output_filename) else os.path.join(base_path, output_filename)

    # 2. Cargar el archivo Excel de entrada y el JSON
    with open(json_path, "r", encoding="utf-8") as jsonfile:
        questions = json.load(jsonfile)

    df_in = pd.read_excel(input_path, nrows=num_rows if num_rows else None)

    # 3. Recorrer las filas del DataFrame con barra de progreso
    df_list = []
    for index, r in tqdm(df_in.iterrows(), total=len(df_in), desc="Procesando comentarios"):
        idx = {i: r[i] for i in index_cols}
        idx['Block'] = txt_column
        comment = r[txt_column]
        
        output = self.classify(
            comment,
            questions["q1"],
            questions["q2"],
            "topic_area",
            eval_cond=["negativity", 1.5],
            index=idx
        )
        df_list += output

    # 4. Concatenar DataFrames
    df_out = pd.DataFrame(df_list)

    # 5. Guardar el DataFrame final
    df_out.to_excel(output_path, index=False)
    print(f"Archivo guardado con exito en: {output_path}")

    return df_out
