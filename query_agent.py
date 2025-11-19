import pandas as pd
from prompts.query_agent_prompts import prompt_extract, prompt_fallback_python, prompt_normalization, prompt_total
from llm import ask_openai
from typing import Union, List, Dict
import io
import re
import contextlib


class QueryAgent:
    def __init__(self):
        pass

    def remove_markdown_syntax(self, text: str) -> str:
        # Remove triple backtick code blocks (```python ... ```)
        text = re.sub(r"```[\s\S]*?```", lambda m: re.sub(r"^```.*\n|```$", '', m.group()), text)

        # Remove inline code (`code`)
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # Remove bold (**text** or __text__)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

        # Remove italic (*text* or _text_)
        text = re.sub(r"\*(.*?)\*", r"\1", text)

        # Remove blockquotes
        text = re.sub(r"^>\s?", '', text, flags=re.MULTILINE)

        text = text.replace("python", "")
        return text.strip()

    def extract_result(self, text: str, pattern: str, opposite=False) -> str:
        position = text.lower().rfind(pattern.lower())
        if position == -1:
            print(f"Cannot find pattern '{pattern}' in '{text}'. Defaulting to '{text}'...")
            return text
        else:
            position += len(pattern)

        if opposite:
            return text[:position].strip()
        return text[position:].strip()

    def filter_table(self, query: str, table: pd.DataFrame) -> Union[tuple[pd.DataFrame, str], tuple[int, str]]:
        # add row index
        try:
            table = table.drop(columns="index")
        except:
            pass

        table.insert(0, "index", range(len(table)))

        # add column index
        table = pd.concat([pd.DataFrame([table.columns.tolist()], columns=table.columns), table], ignore_index=True)
        table.columns = range(len(table.columns))

        prompt_extract_filled = prompt_extract.format(question=query, table=table.to_html(index=False))
        response = ask_openai([
            {
                "role": "system",
                "content": prompt_extract_filled,
            }
        ])

        rows_columns_extracted = self.remove_markdown_syntax(self.extract_result(response, "Final answer:"))

        try:
            rows_columns_extracted = eval(rows_columns_extracted)
        except:
            print(f"Formatting error while extracting the row and column indices: '{rows_columns_extracted}'")
            return -1, response

        table.columns = table.columns.astype(str)
        columns = [el for i, el in enumerate(table.iloc[0, :]) if i in rows_columns_extracted["columns"]]

        table = table[table["0"].isin(rows_columns_extracted["rows"])]# table["0"] is "index"
        table = table[[str(el) for el in rows_columns_extracted["columns"]]]

        # clean
        table.columns = columns
        try:
            table = table.drop(index=0).reset_index(drop=True)
        except:
            table = table.reset_index(drop=True)

        try:
            table = table.drop(columns='index')
        except:
            pass

        return table, response

    def execute(self, python_text: str, question: str, content: str):
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        error = False
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                exec(python_text.strip())
            except Exception as e:
                error = True
                print("Generated python function is not executable. Falling back to cot...")
                print(e)

        if error:
            result = ask_openai([
                {
                    "role": "system",
                    "content": prompt_fallback_python.format(question=question, content=content),
                }
            ])
        else:
            result = stdout_buffer.getvalue()

        # in case error is True, there's no need to apply the last step "final answer:" after python execution, because it is already done during the error handling
        # so inside the caller function, do not launch the final LLM call if error is True
        return result, error

    def table_normalization(self, query: str, intermediate_tables: dict[str, list[pd.DataFrame]]) -> str:
        html_tables = []
        for sector_key in intermediate_tables.keys():
            for key, tables in intermediate_tables[sector_key].items():
                for table in tables:
                    html_tables.append(table.to_html(index=False))

        prompt_text = "\n\n".join(html_tables)
        prompt = prompt_normalization.format(question=query, tables=prompt_text)
        list_of_rules_raw = ask_openai([
            {
                "role": "system",
                "content": prompt,
            }
        ])

        list_of_rules = self.remove_markdown_syntax(self.extract_result(list_of_rules_raw, "Final answer:"))
        return list_of_rules, list_of_rules_raw

    def table_insertion(self, texts: list[str], tables: Dict[Union[int, str], List[pd.DataFrame]]) -> List[str]:
        """
        Inserts the corresponding HTML tables into the “texts” strings.
        “tables” can be:
          - a dict with contiguous integer keys (0..n-1) corresponding to the indices of texts, or
          - a dict with arbitrary keys (e.g. sector names or filenames). In this case, the order of list(tables.values()) is used.
        If the number of elements in texts and tables does not match, the function makes a best-effort attempt:
          - it uses min(len(texts), len(tables_list)) and does not touch the excess texts.
        """
        # Normalise tables in a list whose i-th entry corresponds to texts[i]
        # Case 1: contiguous numeric keys 0..n-1
        try:
            numeric_keys = False not in [all(isinstance(k, int) for k in tables[section_key].keys()) for section_key in tables.keys()]
        except Exception:
            numeric_keys = False

        keys_sorted = sorted([subkey for subdict in tables.values() for subkey in subdict.keys()])
        dict_wo_subdicts = {subkey:subvalue for subdict in tables.values() for subkey, subvalue in subdict.items()}

        if numeric_keys:
            # Verify that the keys are contiguous starting from 0
            if keys_sorted == list(range(len(keys_sorted))):
                tables_list = [dict_wo_subdicts[i] for i in range(len(keys_sorted))]
            else:
                # Non-contiguous: we still transform them into an ordered list by ascending key
                tables_list = [dict_wo_subdicts[k] for k in keys_sorted]
        else:
            # Non-numeric keys: use insertion order / values()
            tables_list = list(dict_wo_subdicts.values())

        new_texts = []
        n = min(len(texts), len(tables_list))
        # For texts beyond n, we leave the original text
        for i, text in enumerate(texts):
            new_text = text
            if i < n:
                tables_for_text = tables_list[i] or []
                for j in range(len(tables_for_text)):
                    # protection: ensure that the placeholder exists in the text before replacing it
                    placeholder = f"<Table{j + 1}>"
                    if placeholder in new_text:
                        # convert the table to html (index=False)
                        try:
                            new_text = new_text.replace(placeholder, tables_for_text[j].to_html(index=False))
                        except Exception as e:
                            print(f"DEBUG: error converting table[{i}][{j}] to html: {e}", flush=True)
                            new_text = new_text.replace(placeholder, "")  # fallback: remove placeholder
                    else:
                        # placeholder not present: we could still hang the table or ignore it; for now, let's ignore it
                        pass
            else:
                print(f"DEBUG: No table available for texts[{i}] - I'll leave the original text.", flush=True)

            new_texts.append(new_text)

        return new_texts

    def query(self, query: str, tables: Dict[str, List[pd.DataFrame]], texts: List[str]) -> str:
        """
        given a query and a list of tables, this function processes each table in this way:
        - Filtering: extraction of relevant rows and columns from each table
        - Table normalization: definition of the rule to change values across different units of measurements. This is done in a single LLM call with all the tables and the query.
        - Table insertion: the tables are re-inserted back into the page text
        - PoT: the LLM generates the Python code to answer the question
        - Python execution: execute the Python code
        - Final answer: the final result is given back to the LLM, which produces a general response explaining the answer
        """

        if all(isinstance(v, list) for v in tables.values()):
            sector_question = False
            tmp_dict = {}
            tmp_dict["fake_sector"] = tables
            tables = tmp_dict
        else:
            sector_question = True

        messages = [
            "To answer the question, let's focus on the following tables. Interesting values are highlighted."
        ]
        intermediate_filtered_idx = {}
        intermediate_tables = {}
        error = False

        # filter table
        for sector_key in tables.keys():
            intermediate_tables[sector_key] = {}
            intermediate_filtered_idx[sector_key] = {}
            for key, list_tables in tables[sector_key].items():
                intermediate_tables[sector_key][key] = []
                intermediate_filtered_idx[sector_key][key] = []
                for table in list_tables:
                    intermediate_tables[sector_key][key].append(table)
                    filtered_table, extract_response = self.filter_table(query, table)
                    if isinstance(filtered_table, int) and filtered_table == -1:
                        error_extraction = True
                    else:
                        error_extraction = False

                    if error_extraction:
                        intermediate_filtered_idx[sector_key][key].append(-1)
                        #intermediate_tables[key].append(-1)
                    else:
                        intermediate_filtered_idx[sector_key][key].append(extract_response)
                        #intermediate_tables[key].append(filtered_table)

        table_txt = ""
        for sector_key in intermediate_tables.keys():
            if sector_question:
                table_txt += f"The company below are inside the following sector: {sector_key}\n\n"
            for key, values in intermediate_tables[sector_key].items():
                table_txt += f"Company name: {key}\n\n"
                for value in values:
                    table_txt += value.to_html(index=False) + "\n\n"
        table_txt = table_txt.strip()
        messages.append("\n\n"+table_txt)

        # normalize table
        list_of_rules, list_of_rules_raw = self.table_normalization(query, intermediate_tables)
        messages.append("\n\n# Normalization\n\n"+list_of_rules_raw)

        # Table insertion
        new_texts = self.table_insertion(texts, intermediate_tables)

        # PoT
        prompt = prompt_total.format(question=query, paragraph="\n\n".join(new_texts) + "\n\n" + list_of_rules)
        python_text_raw = ask_openai([
            {
                "role": "system",
                "content": prompt,
            }
        ])
        messages.append("\n\n# 🧠 Program of Thought\n\n"+self.remove_markdown_syntax(self.extract_result(python_text_raw, "Final answer:", opposite=True))+"\n"+self.extract_result(python_text_raw, "Final answer:"))
        python_code = self.remove_markdown_syntax(self.extract_result(python_text_raw, "Final answer:"))
        results, error = self.execute(python_code, query, '\n\n'.join(new_texts) + "\n\n" + list_of_rules)

        if error:
            return "PoT execution failed. Falling back to CoT...\n" + results, None

        # Final answer
        # results = self.remove_markdown_syntax(self.extract_result(results, "Final answer:"))
        messages.append("\n\nFinal response: "+results)
        return "".join(messages), intermediate_filtered_idx
