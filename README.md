SKELETON.PY – MOCK COGNITIVE-LOAD SCORER
========================================

## Overview

`skeleton.py` implements a small regressor that maps a single assistant response (a text string) to a numeric *cognitive-load* proxy.

- **Current version:**  
  Labels are derived from `textstat` readability scores (no external data needed).

- **Final version:**  
  Labels will be replaced by **ELO-style scores** learned from many pairwise “battles” between responses.

Other scripts (e.g. `scorer_conv.py`) treat this module as a black-box oracle via:

```python
predict_cognitive_load(text: str) -> float

Main functions
EXAMPLES

    Small list (~20) of synthetic assistant responses.

    Acts as a stand-in for Dataset A (“response + score”) so we can train without real data.

avg_word_length(text)

    Splits text into words and returns the average word length.

    Simple proxy for how dense / complex the wording is.

extract_features(text)

Returns a dict with:

    avg_word_len

    flesch_reading_ease

    automated_readability_index

    fk_grade (used as label in the mock setup)

During training, only the first three are used as features; fk_grade is the target.
For now, the text “labels itself” via readability indices.
FEATURE_COLUMNS

List of feature names used for training and inference:

["avg_word_len", "flesch_reading_ease", "automated_readability_index"]

Keeps feature ordering consistent across the codebase.
train_model_full()

    Trains a RandomForestRegressor on all EXAMPLES using the features above and fk_grade as target.

    Used to produce the final model that will be saved and reused.

save_model(model_path) / load_model(model_path)

    save_model: trains train_model_full() and saves {model, feature_cols} to a .joblib file.

    load_model: loads that file and returns (model, feature_cols).

predict_cognitive_load(text, ...)

    Public interface for the rest of the pipeline.

    Loads the model (if needed), extracts features for text, builds the feature vector, and returns a single float score.

    scorer_conv.py calls this for each assistant turn and averages the scores per strategy (direct / hedging / clarify).

Command-line usage

    python skeleton.py --eval → quick train/test, metrics, feature importances.

    python skeleton.py --save-model fk_rf.joblib → train on all examples and save the model.

How this will change in the final version

Core idea: keep the interface the same (predict_cognitive_load(text) -> float) and only change internals.
Labels (fk_grade → ELO scores) –> Most important change

    Now: label y is fk_grade computed from textstat.

    Later: EXAMPLES will be replaced by a real Dataset A with "score" columns coming from an ELO / Bradley-Terry system (pairwise battles between responses).

    In train_model_full() (and main()), y will be that ELO score, not a readability index.

Features (extract_features)

    Now: a few readability metrics + average word length.

    Later: extended with more features (extra readability metrics, token/sentence counts, punctuation ratios, maybe embedding-based signals) plus the chosen metric.

    FEATURE_COLUMNS will be updated, and training/prediction will automatically use the richer feature set.

    Teammates working on feature engineering edit extract_features and FEATURE_COLUMNS.

Model type (RF now, GBM / NN later)

    Now: fixed RandomForestRegressor.

    Later: could be tuned RF (more n_estimators, grid search), Gradient Boosting, XGBoost, LightGBM, or a small neural net.

    As long as train_model_full() returns an object with .predict(X) and we still save {model, feature_cols} via joblib, the rest of the pipeline (including scorer_conv.py and the selector network in script 3) does not need to change.

SCORER_CONV.PY – SCORING CONVERSATION STRATEGIES
Overview

scorer_conv.py takes mock conversations in the “B-shape” format and uses the trained model from skeleton.py to assign a cognitive-load score to each strategy:

    direct

    hedging

    clarify

It then decides which strategy is best (lowest cognitive load) for each item and writes the result to scored_dataset_B.json.

    Current version:

        Conversations are invented examples generated for testing.

        The cognitive-load model comes from skeleton.py, trained on readability features from textstat.

    Final pipeline:

        Conversations will come from Task 2.

        The model from skeleton.py will be replaced by a more powerful scorer (e.g. tuned RandomForest, better features, or a different regressor).

Input data format – B-shaped conversations

The input file is a JSON list where each element has one base_conv plus three alternative strategy branches:

    direct

    hedging

    clarify

Each of those is an OpenAI-style JSON conversation with a "messages" array, e.g.:

{
  "id": 1,
  "base_conv": {
    "messages": [
      {
        "role": "user",
        "content": "How do I reset my home router?"
      }
    ]
  },
  "direct": {
    "messages": [
      {
        "role": "assistant",
        "content": "Press and hold the reset button for 10–15 seconds until the lights flash."
      }
    ]
  },
  "hedging": {
    "messages": [
      {
        "role": "assistant",
        "content": "It can vary by model, but generally you hold the recessed reset button for about 10–15 seconds."
      }
    ]
  },
  "clarify": {
    "messages": [
      {
        "role": "assistant",
        "content": "Which brand/model is your router? Steps differ by vendor and firmware."
      },
      {
        "role": "user",
        "content": "It’s a TP-Link Archer C6."
      },
      {
        "role": "assistant",
        "content": "For that model, hold reset/WPS ~10 seconds until LEDs blink, then wait for reboot."
      },
      {
        "role": "user",
        "content": "Will that erase my settings?"
      },
      {
        "role": "assistant",
        "content": "Yes, a factory reset clears settings. For a soft restart, power-cycle instead."
      }
    ]
  }
}

In the final pipeline, these JSON items will come directly from Task 2, which will generate such conversations in bulk.
What the script does

For each item in the input JSON list:

    Load the trained model from skeleton.py

        Uses importlib to import skeleton.py.

        Uses load_model(model_path) or calls predict_cognitive_load directly.

    Score each strategy

        For every assistant message in direct, hedging, and clarify, call the model from Part 1 (predict_cognitive_load).

        For each strategy, take the mean of the per-turn scores to get the final strategy score.

    Write the scored dataset

        Output example:

        {
          "id": 1,
          "base_conv": { "messages": [...] },
          "direct": 1.3,
          "hedging": 2.1,
          "clarify": 4.5,
          "best": "clarify"
        }

        direct, hedging, clarify now hold numeric scores (floats).

        The scores are calculated using the model from Part 1, then averaged (mean) per response.

        best is the label used later by the selector network (script 3) as the target class.

        This file is the bridge between “lots of raw conversations” and a “clean supervised dataset” for learning a policy.

Final version

Conceptually, the final version of this script should look the same from the outside:

    Input: a JSON file of conversations with {base_conv, direct, hedging, clarify} in the Task 2 format.

    Core logic: for each assistant message, call the cognitive-load model and aggregate per strategy.

    Output: a scored_dataset_B.json file with numeric scores per strategy + a best label.

The differences will be inside the black boxes:
Conversations (input)

    Now: generated by a small helper script and manually designed seeds; synthetic and small.

    Later: produced by Task 2, using:

        a larger set of user questions,

        a real LLM to generate direct / hedging / clarify variants,

        more realistic multi-turn structures.

Cognitive-load model (skeleton.py)

    Now: RandomForest with a few textstat features, using readability as a proxy (fk_grade).

    Later:

        Same API (predict_cognitive_load(text) -> float), but:

            trained on real ELO-style scores from many pairwise battles,

            with more features (richer textstat features, maybe embeddings, etc.),

            potentially a more complex regressor (more n_estimators, tuned hyperparameters, or a different model class).

        You might run a grid search or other hyperparameter search to find a better model before saving it.

TRAIN_SELECTOR_TINY.PY – STRATEGY SELECTOR
Overview

train_selector_tiny.py trains a small neural network that learns to choose the best response strategy (direct, hedging, or clarify) based only on the initial user message.

It uses as input the file scored_dataset_B.json created by script 2, where each item already has:

    numeric scores for direct, hedging, clarify (from the cognitive-load model), and

    a "best" field with the strategy that has the lowest predicted cognitive load.

Data flow

For each item in scored_dataset_B.json:

    Take the first user message from base_conv["messages"].

    Encode this text with a SentenceTransformer → embedding vector z.

    Map "best" to a class label:

        direct → 0

        hedging → 1

        clarify → 2

So the training data is a set of (embedding, label) pairs.
Model and training

    Model: tiny MLP classifier in PyTorch

        Input: sentence embedding.

        Hidden layer: small number of units (e.g. 64) with ReLU + dropout.

        Output: 3 logits (one per strategy).

    Training:

        Train/validation split (with safe fallback if the dataset is very small).

        Cross-entropy loss, Adam optimizer, a few epochs.

        Prints training loss, optional validation accuracy, and example predictions.

Inference (intended use)

After training, the selector can be used as:

    Take a new user query.

    Embed it with the same SentenceTransformer.

    Pass the embedding through the MLP.

    Use argmax over the 3 outputs to pick direct, hedging, or clarify.

How this changes in the final version

The structure stays the same:

    user text → embedding → selector model → best strategy

Final version differences:

    The model may be slightly larger or better tuned, but still small (we do not expect huge datasets, so too many parameters are not useful).

    We may swap to a different sentence-embedding model if needed.

    Training hyperparameters (epochs, learning rate, hidden size) can be adjusted once we see real data.

The input format, the use of the "best" label from script 2, and the overall role of this script (learn to predict the best strategy from the initial user message) remain the same.
