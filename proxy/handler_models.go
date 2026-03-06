package proxy

import (
	"net/http"
)

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	models, err := s.chain.ListModels(ctx)
	if err != nil {
		writeAPIError(w, NewInternalError(err.Error()))
		return
	}

	writeJSON(w, http.StatusOK, FormatModelsResponse(models))
}
