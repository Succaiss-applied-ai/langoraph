package langoraph

import "fmt"

func recoveredPanicError(scope string, recovered any) error {
	if recovered == nil {
		return nil
	}
	return fmt.Errorf("%s panic: %v", scope, recovered)
}
