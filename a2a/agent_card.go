package a2a

// AgentCard is the public metadata document that describes an agent's
// capabilities, skills, and endpoint. It is served at /.well-known/agent.json
type AgentCard struct {
	Name               string               `json:"name"`
	Description        string               `json:"description,omitempty"`
	URL                string               `json:"url"`
	Version            string               `json:"version"`
	Capabilities       AgentCapabilities    `json:"capabilities"`
	Skills             []AgentSkill         `json:"skills"`
	DefaultInputModes  []string             `json:"defaultInputModes,omitempty"`
	DefaultOutputModes []string             `json:"defaultOutputModes,omitempty"`
	Provider           *AgentProvider       `json:"provider,omitempty"`
	DocumentationURL   string               `json:"documentationUrl,omitempty"`
	Authentication     *AgentAuthentication `json:"authentication,omitempty"`
}

// AgentCapabilities describes what features an agent supports
type AgentCapabilities struct {
	Streaming         bool `json:"streaming"`
	PushNotifications bool `json:"pushNotifications"`
	StateTransitions  bool `json:"stateTransitions"`
}

// AgentSkill describes a specific capability or skill of an agent
type AgentSkill struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	Tags        []string `json:"tags,omitempty"`
	Examples    []string `json:"examples,omitempty"`
}

// AgentProvider describes the organization providing the agent
type AgentProvider struct {
	Organization string `json:"organization"`
	URL          string `json:"url,omitempty"`
}

// AgentAuthentication describes the authentication schemes supported by the agent
type AgentAuthentication struct {
	Schemes []string `json:"schemes"` // e.g. "bearer", "none"
}
