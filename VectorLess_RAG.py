from langchain_community.document_loaders import PyPDFLoader
from rank_bm25 import BM25Okapi
loader= PyPDFLoader(r"C:\Users\HP\Desktop\Small_Business_Employee_Policy.pdf")
documents= loader.load()

pages= {}
for i,doc in enumerate(documents):
    pages[i+1]= doc.page_content
#print(pages)

import re
graph= {}
for page_num, text in pages.items():
    ref= re.findall(r"page\s+ (\d+)", text.lower())
    policy_ref= re.findall(r"(?:refer to|see)\s+the\s+(.*?)\s+policy",text.lower())
    section_ref= re.findall(r"(?:refer to|see)\s+the\s+(.*?)\s+section",text.lower())
    appendix_ref= re.findall(r"(?:refer to|see)\s+appendix\s+([a-z])", text.lower())
    chapter_ref= re.findall(r"(?:refer to|see)\s+chapter\s+(\d)",text.lower())

    # print("Page:", page_num)
    # print("section_refs =", section_ref)

    graph[page_num]= {
        "pages": [int(x) for x in ref],
        "sections":  section_ref,
        "policies": policy_ref,
        "appendices": appendix_ref,
        "chapters": chapter_ref
        }
#print(graph.get(25, {}))

tokenized_pages= []
page_numbers= []
for page_num, text in pages.items():
    tokenized_pages.append(text.lower().split())
    page_numbers.append(page_num)
bm25= BM25Okapi(tokenized_pages)


query= input("Question: ")
scores= bm25.get_scores(query.lower().split())
best_index= scores.argmax()
best_page= page_numbers[best_index]


context_pages= {best_page}
# Direct page references
for p in graph[best_page]["pages"]:
    context_pages.add(p)

all_refs = (
    graph[best_page]["sections"]
    + graph[best_page]["policies"]
    + graph[best_page]["chapters"]
    + graph[best_page]["appendices"]
)
for ref in all_refs:

    for page_num, text in pages.items():
        if ref in text.lower():
            context_pages.add(page_num)


context= ""
for page in context_pages:
    context+= "\n\n"
    context+= f"PAGE {page}\n"
    context+= pages[page]
#print(context)


prompt= f"""
Use the context below to answer.

{context}


Question:
{query}
"""

from groq import Groq
client= Groq(api_key="groq api key")
response= client.chat.completions.create(model= "llama-3.3-70b-versatile",
                                         messages= [{"role": "user", "content": prompt}])
answer = response.choices[0].message.content

print("\nPages used:", sorted(context_pages))
print("\n", answer)