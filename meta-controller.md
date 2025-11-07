# Initialize with any meta-learning algorithm
meta_learner = MetaLearner(
    model=your_model,
    config=MetaLearnerConfig(...),
    meta_learning_type=MetaLearningType.MAML  # or any other type
)

# Meta-train
for task in tasks:
    loss = meta_learner(support_set, query_set)
    meta_learner.meta_update(loss)

# Adapt to new tasks
fast_weights = meta_learner.adapt(new_support_set)
predictions = meta_learner.predict(new_query_inputs, fast_weights)