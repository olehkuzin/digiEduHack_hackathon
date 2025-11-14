from agent import call_agent


def create_prompt(source_features: list[str]) -> str:
    """
    Build a system prompt instructing the model to map a target feature
    to one of the provided source feature names.

    Parameters
    ----------
    source_features : list[str]
        List of feature names available for matching.

    Returns
    -------
    str
        A formatted prompt containing task rules and the source feature list.
    """
    # Embed each feature inside <feature> tags
    features_xml = ''.join(f'<feature>{sf}</feature>' for sf in source_features)

    return f"""
<task>
    You must determine whether a target feature logically corresponds to any feature from the given list.
    You will receive the target feature description separately.
    If a match exists, output the exact source feature name.
    If no match exists, output exactly: NAN.
</task>

<rules>
    <rule>Respond with only one token: a feature name from the list OR NAN.</rule>
    <rule>No explanations, no comments, no reasoning.</rule>
    <rule>No punctuation or extra text.</rule>
    <rule>Decision must be based ONLY on the provided target feature description.</rule>
</rules>

<input>
    {{TARGET_BLOCK}}
    <source_features>
        {features_xml}
    </source_features>
</input>

<output>
</output>
"""


def create_user_message(target_feature: str, target_feature_values) -> str:
    """
    Create the user message block containing the target feature
    and its values.

    Parameters
    ----------
    target_feature : str
        Name of the feature to classify.
    target_feature_values : Any
        Values associated with the target feature, typically numeric or categorical.

    Returns
    -------
    str
        The XML representation of the target feature description.
    """
    return f"""
<target_feature>
    <name>{target_feature}</name>
    <values>{target_feature_values}</values>
</target_feature>
"""


def map_feature(target_feature: str, target_feature_values, source_features: list[str]):
    """
    Determine whether the target feature corresponds to any source feature
    by sending a structured prompt to the model.

    Parameters
    ----------
    target_feature : str
        Feature name to be mapped.
    target_feature_values : Any
        Data describing the target feature.
    source_features : list[str]
        List of all possible feature names to match against.

    Returns
    -------
    str
        Either a matched feature name or 'NAN'.
    """
    # Construct full system prompt and user message
    prompt = create_prompt(source_features)
    user_message = create_user_message(target_feature, target_feature_values)

    # Query the model and return its one-token decision
    return call_agent(prompt, user_message)
