from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import os

os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOllama(
    model = "llama3.1",
    base_url = "http://localhost:11434")

professor = Agent(role = "Professor de Matemática",
                      goal = """Promove soluções aos alunos com perguntas relacionadas a matemática e lhes dão a resposta.""",
                      backstory = """Você é um excelente professor de matemática que gosta de resolver questões de matemática de uma forma que todos possam entender sua solução""",
                      allow_delegation = False,
                      verbose = True,
                      llm = llm)

task_professor = Task(description="""Responder as perguntas do aluno""",
             agent = professor,
             expected_output="Uma explicação por etapas")


aluno = Agent(role = "Aluno de matemática",
              goal = "Conseguir entender a demonstração do triângulo de pitágoras",
              backstory="""Você é um aluno com muita dificuldade de conhecimento, precisa de uma explicação didática e cheia de analogias, passo a passo""",
              allow_delegation = False,
              verbose=True,
              llm=llm)

task_aluno = Task(description="Fazer perguntas simples ao professor para evoluir o conhecimento até a demonstração do teorema de pitágoras",
                  agent=aluno,
                  expected_output="Uma pergunta")

crew = Crew(
            agents=[professor, aluno],
            tasks=[task_professor, task_aluno],
            languages="Brazilan Portuguese",
            verbose=5
        )

result = crew.kickoff()

print(result)