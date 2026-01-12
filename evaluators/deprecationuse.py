
import evaluator

class deprecationuse(evaluator.Evaluator):
  def __init__(self):
    self.stage = 'compiler_output'

  def Evaluate(self, compiler_output: str, problem_definition: dict):
    '''Checks if deprecated functions are used in the source code'''
    if compiler_output.lower().find('deprecated') > -1:
      return (1, 'Found use of deprecated functionality\n')
    else:
      return (0, '')