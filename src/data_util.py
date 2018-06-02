from config import Config
from enum import Enum

class Speaker(Enum):
    STUDENT = 1
    PLATFORM = 2

def is_student_text(row):
    return row.sent_from in Config.STUDENT_SPEAKERS

def is_platform_text(row):
    return row.sent_from in Config.PLATFORM_SPEAKERS

def is_tutor_question(row):
    return row.sent_from in Config.TUTOR_SPEAKERS and '?' in row.text

def get_speaker(row):
    if is_student_text(row):
        return Speaker.STUDENT
    elif is_platform_text(row):
        return Speaker.PLATFORM
    else:
        raise ValueError("%s is not a recognized speaker" % row.sent_from)


class IndexedRow:
    def __init__(self, index, row, duration=None):
        self.index = index
        self.row = row
        self.duration = duration

class Session(object):
    def __init__(self, session_id, rows):
        self.id = session_id
        self.rows = rows

    def iter_turns(self, start_row=0, num_turns=0, direction=1, concatenator=None):
        assert direction in [1, -1]
        last_row = self.rows.shape[0] - 1
        assert start_row >=0 and start_row <= last_row

        # Only implementing backwards direction for now
        if direction != -1:
            raise NotImplementedError

        row = self.rows.iloc[start_row]
        # Check if there's additional context and we should move start index forward
        turn_speaker = get_speaker(row)
        turn_end = next((i for i in range(start_row, last_row+1) if get_speaker(self.rows.iloc[i]) != turn_speaker), start_row)


        while turn_end > 0 and num_turns > 0:
            turn_speaker = get_speaker(self.rows.iloc[turn_end-1])
            turn_start = next((i + 1 for i in range(turn_end - 1, -1, -1) if get_speaker(self.rows.iloc[i]) != turn_speaker), 0)
            turn = self.rows.iloc[turn_start]

            text = []
            for _, row in self.rows.iloc[turn_start:turn_end].iterrows():
                text.extend(row.text)
                if concatenator is not None:
                    text.append(concatenator)
            if concatenator is not None and text[-1] == concatenator:
                text = text[:-1]

            turn.at["text"] = text
            turn.at["sent_from"] = turn_speaker.name.lower()
            turn_end = turn_start
            num_turns -= 1
            yield turn

        raise StopIteration

    def iter_question_and_response(self, concatenator=None):
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
            start = next((i + 1 for i in range(question_index,-1,-1) if not is_platform_text(self.rows.iloc[i])), question_index)
            end = next((i for i in range(question_index, last_index + 1,1) if not is_platform_text(self.rows.iloc[i])), response_index + 1)
            question_text = []
            for _, row in self.rows.iloc[start:end].iterrows():
                question_text.extend(row.text)
                if concatenator is not None:
                    question_text.append(concatenator)

            if concatenator is not None and question_text[-1] == concatenator:
                question_text = question_text[:-1]
            question.text = question_text
            duration = (self.rows.iloc[end - 1].created_at - self.rows.iloc[start].created_at).seconds

            prev_i = response_index
            yield IndexedRow(question_index, question, duration), IndexedRow(response_index, response)
        raise StopIteration

def get_sessions(data):
    session_ids = data.session_id.unique()
    return [Session(session_id, data.loc[data.session_id == session_id].reset_index()) for session_id in session_ids]

if __name__ == "__main__":
    import cProfile, data_readers
    data = data_readers.read_corpus("dev")
    sessions = get_sessions(data)
    cProfile.run("test(sessions)", filename="profile")
