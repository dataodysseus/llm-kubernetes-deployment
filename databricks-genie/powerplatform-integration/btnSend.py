If(
    IsBlank(Trim(txtQuestion.Text)),
    Notify("Please enter a question", NotificationType.Warning),

    With(
        { _q: Trim(txtQuestion.Text) },
        // show loading
        Set(varIsLoading, true);

        // add user's message
        Collect(
            colConversation,
            {
                Type: "User",
                Message: _q,
                Timestamp: Now()
            }
        );

        // call Genie (pass optionals as a record in the 3rd arg)
        Set(
            varResponse,
            GenieAPIService.Querygeniegeniequerypost(
                varUserId,            // user_id
                _q,                   // question
                {
                    session_id: Coalesce(varSessionId, Blank()),
                    max_wait: 120
                }
            )
        );

        // if we got an answer, append
        If(
            !IsBlank(varResponse.answer_text),
            // keep session id for continuity
            If(!IsBlank(varResponse.session_id), Set(varSessionId, varResponse.session_id));

            Collect(
                colConversation,
                {
                    Type: "Genie",
                    Message: varResponse.answer_text,
                    Timestamp: Now(),
                    SQL: varResponse.sql,
                    RowCount: varResponse.row_count
                }
            );

            // reset input and scroll to bottom
            Reset(txtQuestion);
            Set(varScrollTo, Last(colConversation)),

            // else show error
            Notify("Error getting response from Genie", NotificationType.Error)
        );

        // hide loading
        Set(varIsLoading, false)
    )
)
