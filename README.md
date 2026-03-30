# 📄 Graduation Project: AI-Powered IT Support Priority Classifier
**Student Name:** Abdullahi Abdirahman Yusuf  
(*Abdalla Hiil*)

---

## 1. Problem Statement & Motivation
In large organizations, IT departments receive hundreds of support tickets daily. Currently, these tickets are sorted manually by a human, which is slow and expensive. If a server goes down (High Priority), it might sit in the inbox for hours before a human sees it.

**The Goal:** To build a Machine Learning "brain" that reads the text of a ticket and immediately says: **"This is High Priority"** or **"This is Low Priority."** This ensures that critical security and networking issues are fixed first.

---

## 2. Dataset & Preprocessing
### 2.1 The Data
I used a dataset of **1,500 IT Support Tickets**. Each ticket has:
1.  **Body:** The text written by the user (e.g., "My VPN is not connecting").
2.  **Priority:** The correct label (High, Medium, or Low).

### 2.2 Preprocessing Steps (Data Cleaning)
To make the AI smart, I performed these steps in Python:
* **De-duplication:** I used `drop_duplicates(keep='first')` to remove repeat tickets so the AI doesn't get "lazy."
* **Normalization:** Converted all text to lowercase and removed extra spaces.
* **Vectorization (TF-IDF):** Computers cannot read English. I used **TF-IDF** to turn words into numbers. I used `ngram_range=(1,2)` so the AI can understand two words together, like "System Crash."
* **Class Balancing:** Because I had many "High" tickets but fewer "Low" tickets, I used `class_weight='balanced'` to make sure the AI treats every category as equally important.

---

## 3. Algorithms & Methodology
I trained **two different algorithms** to compare which one is better for an IT environment:

### 3.1 Random Forest (The Winner)
This is an "Ensemble" algorithm that uses many Decision Trees to vote on the result. It is very good at finding patterns in technical words.
* **Why I chose it:** It handles "noisy" data very well.

### 3.2 Logistic Regression (The Baseline)
This is a mathematical model that tries to find a straight line between categories. 
* **Why I chose it:** It is the industry standard for fast text classification.

---

## 4. Results & Discussion
### 4.1 Performance Table
| Metric | Random Forest | Logistic Regression |
| :--- | :--- | :--- |
| **Accuracy** | **59.33%** | 57.00% |
| **Precision** | 0.58 | 0.55 |
| **F1-Score** | 0.59 | 0.56 |

### 4.2 Sanity Checks (Testing)
I tested the model with 3 real-life examples to prove it works:
1.  "Screen is black" $\rightarrow$ Predicted: **High** ✅
2.  "Password reset" $\rightarrow$ Predicted: **Medium** ✅
3.  "Move my desk" $\rightarrow$ Predicted: **Low** ✅

**Conclusion:** Random Forest is the best model for this project. Even though 59% sounds low, in a 3-category problem, it is much higher than a random guess (33%).

---
