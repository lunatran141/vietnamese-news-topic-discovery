\# Contributing Guide



\## Branches



\- main: stable and demo-ready code.

\- develop: integration branch for team development.

\- feature/<task-id>-<short-name>: branch for each task.



Do not push directly to main or develop.



Examples:



\- feature/M01-tfidf-pipeline

\- feature/M04-nmf-pipeline

\- feature/M07-lda-pipeline

\- feature/M10-sentence-embeddings

\- feature/M13-bertopic-baseline

\- feature/S04-docker-skeleton



\## Workflow



1\. Switch to develop.

2\. Pull the latest changes.

3\. Create a feature branch from develop.

4\. Implement and test the task.

5\. Commit using the team convention.

6\. Push the feature branch.

7\. Create a Pull Request into develop.

8\. Ask at least one member to review.

9\. Merge only after review and required checks pass.



\## Git Commands



Before starting a new task:



git switch develop

git pull origin develop

git switch -c feature/<task-id>-<short-name>



After implementing the task:



git status

git add .

git commit -m "feat(M01): add TF-IDF pipeline"

git push -u origin feature/M01-tfidf-pipeline



\## Commit Convention



Format:



<type>(<task-id>): <short description>



Suggested types:



\- feat: add a feature

\- fix: fix a bug

\- docs: update documentation

\- test: add or update tests

\- refactor: restructure code without changing behavior

\- chore: configuration or maintenance



Examples:



\- feat(M01): add TF-IDF pipeline

\- fix(M05): handle empty documents

\- docs(S03): add Git workflow

\- test(P10A): add API health test



\## Code Rules



\- Use English identifiers.

\- Use snake\_case for variables and functions.

\- Use PascalCase for classes.

\- Use UPPER\_SNAKE\_CASE for constants.

\- Add type hints to public functions where appropriate.

\- Add docstrings to public functions and important logic.

\- Do not use unclear names such as data1, tmp2, or final\_final.

\- Do not use except: pass.

\- Do not hard-code personal file paths.

\- Keep configuration values in configs or function parameters.



\## Data Rules



\- Do not overwrite data/raw.

\- Keep article\_id through the full pipeline.

\- Do not change the shared schema without team agreement.

\- Do not use test data for model tuning.

\- Store experiment seed and configuration.



Shared article fields:



\- article\_id

\- title

\- content

\- published\_at

\- source

\- url

\- category\_gold, when available



\## Security Rules



Never commit:



\- Passwords

\- API keys

\- Access tokens

\- .env files

\- Personal absolute paths

\- Private or restricted datasets



Use .env.example to document required environment variables without real secrets.



\## Output Rules



Model runs should use the shared output contract when applicable:



\- model/

\- config.json

\- document\_topics.parquet

\- topic\_keywords.json

\- representative\_docs.json

\- run\_metadata.json



Use:



\- artifacts/ for machine-consumed outputs.

\- results/ for metrics, tables, figures, runtime, and error analysis.



\## Pull Request Requirements



Each Pull Request must include:



\- Task ID

\- Summary of changes

\- Commands used to run or test

\- Test results

\- Artifact or result location

\- Notes for the reviewer



Important work requires at least one reviewer before merging into develop.

