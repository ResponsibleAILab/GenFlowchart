# GenFlowchart: Parsing and Understanding Flowcharts Using Generative AI

Flowcharts serve as integral visual aids, encapsulating both logical flows and specific component-level information in a manner easily interpretable by humans. However, automated parsing of these diagrams poses a significant challenge due to their intricate logical structure and text-rich nature.

In this paper, we introduce GenFlowChart, a novel framework that employs generative AI to enhance the parsing and understanding of flowcharts. First, a cutting-edge segmentation model is deployed to delineate the various components and geometrical shapes within the flowchart using the Segment Anything Model (SAM). Second, Optical Character Recognition (OCR) is utilized to extract the text residing in each component for deeper functional comprehension. Finally, we formulate prompts using prompt engineering for the generative AI to integrate the segmented results and extracted text, thereby reconstructing the flowchart's workflows. To validate the effectiveness of \modelname, we evaluate its performance across multiple flowcharts and benchmark it against several baseline approaches.

## Installation

Install the following libraries before executing the commands as shown in the provided Jupyter notebook:

```bash
pip install pdf2image PyMuPDF 
pip install pytesseract
pip install bert_score
pip install -U sentence-transformers
pip install openai
pip install Word2Vec
