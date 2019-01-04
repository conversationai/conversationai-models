"""Sets up and start the Dataflow job for data preparation."""

import argparse
import logging
import os
import sys

import apache_beam as beam
import configparser
from preprocessing import preprocessing


def _parse_arguments(argv):
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description='Runs Preprocessing on Civil comments data.')
  parser.add_argument(
      '--cloud', action='store_true', help='Run preprocessing on the cloud.')
  parser.add_argument('--job_name', required=False, help='Dataflow job name')
  parser.add_argument(
      '--job_dir',
      required=True,
      help='Directory in which to stage code and write temporary outputs')
  parser.add_argument(
      '--output_folder',
      required=True,
      help='Directory where to write train, eval and test data')
  parser.add_argument('--input_data_path')
  parser.add_argument(
      '--oversample_rate',
      required=False,
      default=5,
      type=int,
      help='How many times to oversample the targeted class')
  args = parser.parse_args(args=argv[1:])
  return args


def _set_logging(log_level):
  logging.getLogger().setLevel(getattr(logging, log_level.upper()))


def _parse_config(env, config_file_path):
  """Parses configuration file.

  Args:
    env: The environment in which the preprocessing job will be run.
    config_file_path: Path to the configuration file to be parsed.

  Returns:
    A dictionary containing the parsed runtime config.
  """
  config = configparser.ConfigParser()
  config.read(config_file_path)
  return dict(config.items(env))


def main():
  """Configures pipeline and spawns preprocessing job."""
  args = _parse_arguments(sys.argv)
  config = _parse_config('CLOUD' if args.cloud else 'LOCAL', 'config.ini')
  options = {'project': str(config.get('project'))}
  if args.cloud:
    if not args.job_name:
      raise ValueError('Job name must be specified for cloud runs.')
    options.update({
        'job_name':
            args.job_name,
        'max_num_workers':
            int(config.get('max_num_workers')),
        'setup_file':
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), 'setup.py')),
        'staging_location':
            os.path.join(args.job_dir, 'staging'),
        'temp_location':
            os.path.join(args.job_dir, 'tmp'),
        'zone':
            config.get('zone')
    })

  pipeline_options = beam.pipeline.PipelineOptions(flags=[], **options)
  _set_logging(config.get('log_level'))
  with beam.Pipeline(
      str(config.get('runner')), options=pipeline_options) as pipeline:
    preprocessing.run_artificial_bias(
        pipeline,
        train_input_data_path=args.input_data_path,
        output_folder=args.output_folder,
        oversample_rate=args.oversample_rate)


if __name__ == '__main__':
  main()
