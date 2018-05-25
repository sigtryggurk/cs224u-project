
def is_student_text(row):
    return row.sent_from == "student"

def is_tutor_text(row):
    return row.sent_from == "tutor"

def is_tutor_question(row):
    return is_tutor_text(row) and '?' in row.text

class IndexedRow:
    def __init__(self, index, row):
        self.index = index
        self.row= row

class Session(object):
    def __init__(self, session_id, rows):
        self.id = session_id
        self.rows = rows

    def iter_question_and_response(self):
        last_index = max(self.rows.index)
        questions = filter(lambda indexed_row: is_tutor_question(indexed_row[1]), self.rows.iterrows())
        prev_i = None
        for question_index, question in questions:
            if question_index == last_index:
                # Skip question if it's the last utterance in the session
                continue

            if prev_i is not None and question_index <= prev_i:
                # Skip candidate if it was a part of the previous tutor question
                continue

            response_index, response = next(filter(lambda indexed_row: is_student_text(indexed_row[1]), self.rows.iloc[question_index+1:].iterrows()), (-1, None))
            if response is None:
                # If there are no responses to a question, we can safely exit
                raise StopIteration

            # Collapse adjacent tutor messages into question text
            start = next((i + 1 for i in range(question_index,-1,-1) if not is_tutor_text(self.rows.iloc[i])), question_index)
            end = next((i for i in range(question_index, last_index + 1,1) if not is_tutor_text(self.rows.iloc[i])), response_index + 1)
            question_text = []
            for _, row in self.rows.iloc[start:end].iterrows():
                question_text.extend(row.text)
            question.text = question_text

            prev_i = response_index
            yield IndexedRow(question_index, question), IndexedRow(response_index, response)
        raise StopIteration

def get_sessions(data):
    session_ids = data.session_id.unique()
    return [Session(session_id, data.loc[data.session_id == session_id].reset_index()) for session_id in session_ids]

if __name__ == "__main__":
    pass
