
import evaluator

class error(evaluator.Evaluator):
  def __init__(self):
    self.stage = 'compiler_output'

  def Evaluate(self, compiler_output: str, problem_definition: dict):
    '''Checks if errors are detected by the compiler'''
    if compiler_output.lower().find('error:') > -1:
      return (1, 'Found errors while compiling\n')
    else:
      return (0, '')