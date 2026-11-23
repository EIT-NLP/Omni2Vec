
def process_query(query, prompt, image_token=''):
    if prompt:
        query = f'{prompt} {query} {image_token}'
    else:
        query = f'{query} {image_token}'
    return query

query_source2prompt = {
    "NeurIPS Papers": "This query is about a research paper from NeurIPS, a leading AI/ML conference. The document contains technical discussions, methodologies, and findings. Identify relevant papers and sections that address the query: ",  # 10,000
    "Textbooks": "This query is related to a college-level textbook, which provides structured explanations, definitions, and examples. Find the most relevant concepts or explanations that address the query: ",    # 5,000
    "ICML Papers": "This query is about a research paper from ICML, a leading AI/ML conference. The document contains theoretical insights, experiments, and applications. Identify relevant papers and sections that best answer the query: ",    # 5,000
    "Manuallib": "This query pertains to a product manual, which contains detailed technical specifications, usage instructions, and troubleshooting steps. Find the most relevant section that answers the query: ",    # 20,000
    "ArxivQA": "This query is related to retrieving a relevant figure from an ArXiv research paper. The retrieved figure should contain scientific plots, mathematical visualizations, or experimental results that best address the query: ",  # 25,856
    "ChartQA": "This query is related to retrieving a relevant chart that visually represents numerical or categorical data. The retrieved chart should contain bar graphs, line charts, or other visual elements necessary to analyze trends, compare values, or extract insights related to the query: ",	  # 4,224
    "MP-DocVQA": "This query is related to retrieving a relevant page from a multi-page document, such as reports, invoices, or research papers. The retrieved document should contain text, tables, or structured information necessary to answer the query: ",	  # 10,624
    "InfoVQA": "This query is related to retrieving an infographic that visually presents statistical or factual information using charts, icons, and structured layouts. The retrieved image should contain the necessary visual elements to provide the best context for answering the query: ",	  # 17,664
    "PlotQA": "This query relates to retrieving a relevant plot or chart that visually represents numerical data. The retrieved figure should contain the necessary information to analyze trends, compare values, or extract key insights related to the query: ",	  # 56,192
    "SlideVQA": "This query is related to retrieving a relevant presentation slide that visually presents structured information. The retrieved slide should contain the necessary text, charts, or graphics to provide the best answer to the query: ",	  # 8,192
}

target_source2prompt = {
    "Textbooks": "A textbook page with structured educational content and explanations.",
    "ICML Papers": "A research paper from ICML, covering machine learning topics.",
    "NeurIPS Papers": "A research paper from NeurIPS on AI and ML topics.",
    "Manuallib": "A product manual page with technical specifications and instructions.",
    "InfoVQA": "An infographic with structured data, charts, and annotations.",
    "PlotQA": "A numerical data visualization, such as bar charts or line graphs.",
    "SlideVQA": "A presentation slide with text, bullet points, and diagrams.",
    "ArxivQA": "A figure from a research paper, including plots or experimental results.",
    "MP-DocVQA": "A page from a multi-page document with text or tables.",
    "ChartQA": "A statistical chart comparing values or analyzing trends.",
}
