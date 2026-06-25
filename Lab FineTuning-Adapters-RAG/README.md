# Lab: Fine-Tuning vs Adapters vs RAG

##  Goal

Understand and compare 3 ways to give knowledge to LLMs:

- Fine-Tuning
- Adapters (LoRA)
- RAG (Retrieval Augmented Generation)

---

##  Project Structure

Fine-Tuning aktualisiert die Gewichte des Modells mit Trainingsdaten, sodass es Wissen oder Verhalten dauerhaft lernt.
Adapter (LoRA) sind eine leichte Form des Fine-Tunings, bei der nur kleine zusätzliche Schichten trainiert werden, während das Hauptmodell eingefroren bleibt.
RAG (Retrieval-Augmented Generation) trainiert das Modell nicht, sondern holt relevante Informationen zur Laufzeit aus externen Dokumenten.
Fine-Tuning verändert das Modell, Adapter erweitern es leicht, und RAG nutzt externes Wissen ohne das Modell zu verändern.