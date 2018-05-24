import pandas as pd
import progressbar

def is_student_text(row):
    return row.sent_from == "student"

def is_tutor_text(row):
    return row.sent_from == "tutor"

def is_tutor_question(row):
    return is_tutor_text(row) and '?' in row.text

def get_questions_and_response_times(data):
    questions = []
    response_times_sec = []
    session_ids = []

    nrows = data.shape[0]
    progress = progressbar.ProgressBar(max_value=nrows).start()

    i = 0
    while i < nrows:
        current_row = data.iloc[i]
        if is_tutor_question(current_row):
            question = current_row.text
            question_time = current_row.created_at
            response_time = None

            # Look back to see if tutor provided previous context
            j = i - 1
            while j >= 0 and is_tutor_text(data.iloc[j]):
                prev_row = data.iloc[j]
                if prev_row.session_id != current_row.session_id:
                    break

                question = data.iloc[j].text + question
                j -= 1

            # Look forward for response and potential additional tutor context
            j = i + 1
            while j < nrows:
                next_row = data.iloc[j]
                if next_row.session_id != current_row.session_id:
                    break

                if is_student_text(next_row):
                    response_time = next_row.created_at
                    break
                elif is_tutor_text(next_row):
                    question += next_row.text
                    #question_time = next_row.created_at
                j += 1

            if len(question) > 1 and question_time is not None and response_time is not None:
                questions.append(question)
                response_times_sec.append((response_time - question_time).seconds)
                session_ids.append(current_row.session_id)
                i = j

        progress.update(i)
        i += 1

    dataset = pd.DataFrame.from_dict({"session_id": session_ids, "question": questions, "response_time_sec": response_times_sec})
    progress.finish()
    return dataset

if __name__ == "__main__":
    import data_readers
    data = data_readers.read_corpus("tiny")
    print(len(get_questions_and_response_times(data)))
