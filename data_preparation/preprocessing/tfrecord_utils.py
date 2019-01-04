"""Utilities to decode and encode TF Records.

These utilities are wrappers around TF-Tranform coders to handle the
    specificities around optional fields.
"""

import apache_beam as beam
from tensorflow_transform import coders


class Schema(object):
  """Defines the dataset schema for tf-transform.

  We should have used dataset_schema from tensorflow_transform.tf_metadata.
      However, there is a lack of support for `FixedLenFeature` default value,
      and an exception is triggered by _feature_from_feature_spec.
  TODO(fprost): Submit internal bug here.
  """

  def __init__(self, spec):
    self._spec = spec

  def as_feature_spec(self):
    return self._spec


class DecodeTFRecord(beam.DoFn):
  """Wrapper around ExampleProtoCoder for decoding optional fields.

  To decode a TF-Record example, we use the  coder utility
    'tensorflow_transform.codersExampleProtoCoder'. For optional fields,
    (indicated by 'default_value' argument for `FixedLenFeature`), the coder
    will generate the default value when the optional field is missing.
  This wrapper post-processes the coder and removes the field if the default
      value was used.
  """

  def __init__(self,
               feature_spec,
               optional_field_names,
               rule_optional_fn=lambda x: x < 0):
    """Initialises a TF-Record decoder.

    Args:
      feature_spec: Dictionary from feature names to one of `FixedLenFeature`,
        `SparseFeature` or `VarLenFeature. It contains all the features to parse
        (including optional ones).
      optional_field_names: list of optional fields.
      rule_optional_fn: function that take the value of an optional field and
        returns True if the value is indicative of a default value (e.g.
        resulting from the default value of parsing FixedLenFeature).  Current
        code requires that all optional_field_names share the rule_optional_fn.
    """
    self._schema = Schema(feature_spec)
    self._coder = coders.ExampleProtoCoder(self._schema)
    self._optional_field_names = optional_field_names
    self._rule_optional_fn = rule_optional_fn

  def process(self, element):
    parsed_element = self._coder.decode(element)
    for identity in self._optional_field_names:
      if self._rule_optional_fn(parsed_element[identity]):
        del parsed_element[identity]
    yield parsed_element


class EncodeTFRecord(beam.DoFn):
  """Wrapper around ExampleProtoCoder for encoding optional fields."""

  def __init__(self, feature_spec, optional_field_names):
    """Initialises a TF-Record encoder.

    Args:
      feature_spec: Dictionary from feature names to one of `FixedLenFeature`,
        `SparseFeature` or `VarLenFeature. It contains all the features to parse
        (including optional ones).
      optional_field_names: list of optional fields.
    """
    self._feature_spec = feature_spec
    self._optional_field_names = optional_field_names

  def process(self, element):
    element_spec = self._feature_spec.copy()
    for identity in self._optional_field_names:
      if identity not in element:
        del element_spec[identity]
    element_schema = Schema(element_spec)
    coder = coders.ExampleProtoCoder(element_schema)
    encoded_element = coder.encode(element)
    yield encoded_element
