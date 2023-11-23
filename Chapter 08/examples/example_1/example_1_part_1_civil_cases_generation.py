from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser

import os
from docx import Document
from typing import List


def generate_civil_cases_names() -> List[str]:
    generate_civil_cases_names_template = """
        You are a lawyer and a legal expert. Generate {number_of_cases} civil cases names.\n{format_instructions}
    """

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=generate_civil_cases_names_template,
        input_variables=["number_of_cases"],
        partial_variables={"format_instructions": format_instructions},
    )

    model = OpenAI(temperature=0.9, max_tokens=4000)
    _input = prompt.format(number_of_cases="80")
    output = model(_input)
    list_of_processes = output_parser.parse(output)

    return list_of_processes


def generate_civil_cases(list_of_processes):
    if not os.path.exists("civil_cases"):
        os.makedirs("civil_cases")

    generate_civil_case_template = """
        You are a lawyer and a legal expert. Generate content of civil case: {civil_case_name}. Include title, full names of parties, background, claims, evidence, legal issues and procedural status.
    """

    llm = OpenAI(temperature=0.9, max_tokens=4000)
    prompt = PromptTemplate(
        input_variables=["civil_case_name"],
        template=generate_civil_case_template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    for i, process in enumerate(list_of_processes, start=1):
        legal_process_content = chain.run(process)
        doc = Document()
        doc.add_paragraph(legal_process_content)
        doc.save(f"civil_cases/civil_case_{i}.docx")


if __name__ == "__main__":
    process_names = generate_civil_cases_names()
    generate_civil_cases(process_names)
