![Studentia - Read.ME Title](images/studentia-banner.png)
# **Chatbot Development History**
This repository is not about the final Chatbot that we used in our app.<br/> 
For the information about the final Chatbot: [Studentia-Chatbot repository](https://github.com/Studentia/Studentia-Chatbot) <br/>
In this repository, you can find how we prepare the data and the various models we experimented with.

## **Table of Contents**

1. Database Preprocessing
    - Data Format & Structure
    - Tokenization Process
    - Installation
    - Source Code
    - Results
2. Models We Tried & Which One We Chose
    - Problem: glove-wiki-gigaword Embedding Model
    - Solution: OpenAI's Curie Embedding Model
    - Key Source Code Sections
    - Other Models We Tried

# **1. Database Preprocessing**

**`tokenizeDBandConvertToCSV.ipynb`**

This section presents code for preprocessing data related to Ewha Womans University, organizing it for OpenAI's GPT-3.5-turbo language model. The goal is to structure the data for efficient embedding, enhancing chatbot performance.

## **Data Format & Structure**

The data is organized into a Python list of dictionaries, each representing an entry. Columns include:

- **title:** Entry title.
- **heading:** Section within each entry.
- **content:** Main textual information.
- **token:** Precalculated token count for efficient embedding.

## **Tokenization Process**

The [tiktoken](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb) library counts tokens using the GPT-3.5-turbo model. The code calculates token counts for each entry's content, adding a "token" column.

#### **How To Install**

```bash
pip install tiktoken
```

#### **How To Test**
Download the **`tokenizeDBandConvertToCSV.ipynb`** file on this repository and run it on Jupyter Notebook or Google Colab.

## **Tokenization**

```python
def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name='gpt-3.5-turbo')
    num_tokens = len(encoding.encode(text=text))
    return num_tokens
```

## **Results**

1. The **`data`** variable represents Studentia’s dataset or sample data.
2. After running the code above, we calculate and add token counts to the dataset.
3. We store the new database in a CSV file (**`ewha_database.csv`**).
4. The database is now ready for embedding and chatbot integration!

![Database Structure Result - Sample Image](images/Screenshot%202023-12-08%20at%202.51.37%20PM.png)

---

# **2. Models We Tried & Which One We Chose**

**`Studentia-Chatbot-Test-1.ipynb`**

Now, we present our documentation of the trial-and-error process, various tests, and model iterations conducted during the development of our closed-domain Q&A chatbot. The initial attempt, labeled **CHATBOT TEST #1**, focused on utilizing the 'glove-wiki-gigaword-100' embedding model. However, this model presented challenges that led to its abandonment in favor of a more suitable alternative.

## **Problem: glove-wiki-gigaword Embedding Model**

The 'glove-wiki-gigaword-100' model, while initially considered, proved unsuitable for our specific case due to the following reasons:

- **Local Installation:** The model required local installation, making it incompatible with the Firebase Functions hosting environment.
- **File Size Limitations:** Even if we wanted to uploaded it to our server, the model's large file size exceeded the upload limitations of Firebase Functions.
- **Limited Efficiency:** Despite being free, the model had limitations, such as a fixed 100 embedding columns and/or floating-point vector embedding size.
    
    ![100 embedding float-point numbers limit - Sample Image](images/Screenshot%202023-12-08%20at%202.51.12%20PM.png)
    

## **Solution: OpenAI's Curie Embedding Model**

To address the challenges posed by the initial model, we transitioned to OpenAI's Embedding Models, specifically 'text-search-curie-doc-001' and 'text-search-curie-query-001.' These models, hosted by OpenAI, offered several advantages:

- **Enhanced Portability:** The models could be used online without the need for local downloads or storage.
- **Increased Efficiency:** With an embedding result of 4096 floating-point numbers, these models provided more room for efficiency and higher similarity.

While acknowledging the associated cost, the advantages of OpenAI's 'Curie' model outweighed the limitations, making it the preferred choice for our specific use case and the overall development of Studentia.

## **Key Source Code Sections**

To illustrate the transition, the initial code for the glove-wiki-gigaword-100 model is included in this repository. The chatbot worked perfectly. However, the definitive and final code, utilizing OpenAI's 'Curie' model, is housed in the [Studentia-Chatbot repository](https://github.com/Studentia/Studentia-Chatbot).

### **Important Source Code Segments & How It Works**

#### **How To Install**

Install the required libraries bu running the following code either on your local terminal or a code editor like Jupyter Notebook or Google Colab.

```bash
%pip install transformers
%pip install gensim
%pip install openai
%pip install pandas
%pip install numpy
%pip install typing
```

#### **How To Test**

Download the **`Studentia-Chatbot-Test-1.ipynb`** file on this repository and run it on Jupyter Notebook or Google Colab.

1. **Embedding Model Initialization**
    
    ```python
    # Import Embedding Library
    import gensim.downloader as api
    
    # Load the pre-trained model glove-wiki-gigaword-100
    word2vec_model = api.load("glove-wiki-gigaword-100")
    ```
    
2. **Compute Word Embeddings for Text**
3. **Compute Document Embeddings**
4. **Order Document Sections by Query Similarity**
5. **Construct Prompt for Chatbot**
6. **Answer Query with Context**

## Other Models We Tried

We also tried other models like global-wiki-gigaword-300 and bert-base-uncased for the embedding process. We did not include the files for these other models as they look exactly the same as the current one only with a few changes to load the model to the system. Everything else remains almost similar.

## __📓 Team 📓__
> Team 11 Cognify <br/>
> 🗓️ 2023.03.27 ~ 2023.12

|[김영애](https://github.com/Y1K5)|[Daniela Fajardo](https://github.com/danielabfajardo)|[Nafisa](https://github.com/NafisaOtakhanova)|
