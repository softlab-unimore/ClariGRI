prompt_extract = """You will be given a question and a table.
ONLY IF there are relevant rows and column, you must indicate the indices of the rows and indices of the columns that could be relevant to answer the question. OTHERWISE If, for a certain table, there are no relevant rows and columns in the table, write an empty list for both "rows" and "columns" keys. You must not try to answer the question, you must only retrieve the relevant rows if there are. You must use the values in the "index" column to refer to the relevant rows.

Additionally, for each selected row include the corresponding row name in the table: use the value from the first non-index column (the column immediately to the right of the "index" column) as the row's name. The "row_names" list must be aligned with the "rows" list (same order). If no such column or name exists for a selected row, use an empty string ('') in the corresponding position of "row_names".

For the column indices, write the number (from left to right starting from 0), not the column name.
First reason step-by-step. Then, write "Final answer: " followed exclusively by a Python dictionary:
{{
    "rows": [row_index1,...,row_indexn],
    "columns": [column_index1,...,column_indexn],
    "row_names": [row_name1,...,row_namen],
}}
The column indices must start from 0.
If there are no relevant rows/columns, return "rows": [] , "columns": [] and "row_names": [].
Ensure that the final answer is in the expected form. Do not write anything else after "Final answer:". Do not use Markdown syntax.

Question: {question}
Table: {table}

Let's think step-by-step. """

prompt_normalization = """Given multiple tables and a question, decide the unit of measure to use for the final answer.
Then, align the table values by converting the needed values to a unique unit of measurement.
If the question specifies the unit of measurement, convert the values to that unit of measurement. Otherwise, you must decide the unit of measurement and convert the values.
You must not rewrite the tables. You must only eventually write a list of rules/formulas indicating how to perform the needed transformations. The transformations must exclusively be about the handling of units of measurement. Do not talk about anything else, including how to solve the question.
A sample rule can be:
1. 1000 meters = 1 kilometer

First reason step-by-step. Then write "Final answer: " followed exclusively by the list of rules/formulas.

Do not try to answer the question, but focus only on the given task. Ensure that the final answer is in the expected form. Do not write anything else after "Final answer:". Do not use Markdown syntax.
Start the response by saying something along the lines of "We need to make sure that the table values, and the question, are aligned in terms of the units of measurement."

Question: {question}
Tables: {tables}

Let's think step-by-step. """

prompt_fallback_python = """
Consider the following question and content. First reason step-by-step, then provide the answer.

Question. {question}
Content: {content}

Let's think step-by-step. 

"""

prompt_total = """
You must create the python code capable of answering the following question given the provided tables. First write your reasoning. Then, in the end, write "Final answer:" followed by the python code and nothing else. The Python code must be runnable "as it is", so make sure to include the relevant imports. At the end of the python function, print the result with print(). If you do not already do so, specify ```python before the Python code and ``` at the end of the Python code.
If the question is boolean, the output must be exclusively a 'yes' or 'no' answer. If the question asks for a list of values, you must answer with a list of values separated with a comma. Write the numerical values with exactly 2 decimal values.
Ensure that the final answer is in the expected form. Do not write anything else after "Final answer:". Do not use Markdown syntax. Write the python code only after "Final answer:", not before.
Start the response by saying something along the lines of "Let's generate the Python code, execute it and output the result.\nLet's think step-by-step.".

Question: {question}
Tables: {paragraph}

Let's think step-by-step.
"""