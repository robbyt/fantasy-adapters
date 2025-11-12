MODULE_DIRS := adk

.PHONY: all
all: help

## help: Display this help message
.PHONY: help
help: Makefile
	@echo
	@echo " Choose a make command to run"
	@echo
	@sed -n 's/^##//p' $< | column -t -s ':' | sed -e 's/^/ /'
	@echo

## test: Run all tests with coverage
.PHONY: test
test:
	@for dir in $(MODULE_DIRS); do \
		echo "Testing $$dir..."; \
		go -C $$dir test -race -cover ./... || exit 1; \
	done

## test-short: Run tests in short mode (fast, no coverage)
.PHONY: test-short
test-short:
	@for dir in $(MODULE_DIRS); do \
		echo "Testing $$dir (short mode)..."; \
		go -C $$dir test -race -short ./... || exit 1; \
	done

## bench: Run performance benchmarks
.PHONY: bench
bench:
	@for dir in $(MODULE_DIRS); do \
		echo "Benchmarking $$dir..."; \
		go -C $$dir test -run=^$$ -bench=. -benchmem ./... || exit 1; \
	done

## lint: Run golangci-lint code quality checks
.PHONY: lint
lint:
	@for dir in $(MODULE_DIRS); do \
		echo "Linting $$dir..."; \
		(cd $$dir && golangci-lint run ./...) || exit 1; \
	done

## lint-fix: Run golangci-lint with auto-fix for common issues
.PHONY: lint-fix
lint-fix:
	@for dir in $(MODULE_DIRS); do \
		echo "Linting and fixing $$dir..."; \
		(cd $$dir && golangci-lint fmt) || exit 1; \
		(cd $$dir && golangci-lint run --fix ./...) || exit 1; \
	done

## tidy: Clean up go modules
.PHONY: tidy
tidy:
	@for dir in $(MODULE_DIRS); do \
		echo "Tidying $$dir..."; \
		go -C $$dir mod tidy || exit 1; \
		go -C $$dir mod verify || exit 1; \
	done

## fmt: Format Go source code
.PHONY: fmt
fmt:
	@for dir in $(MODULE_DIRS); do \
		echo "Formatting $$dir..."; \
		go -C $$dir fmt ./... || exit 1; \
	done

## vet: Run go vet static analysis
.PHONY: vet
vet:
	@for dir in $(MODULE_DIRS); do \
		echo "Vetting $$dir..."; \
		go -C $$dir vet ./... || exit 1; \
	done

## build: Build the library (check for compile errors)
.PHONY: build
build:
	@for dir in $(MODULE_DIRS); do \
		echo "Building $$dir..."; \
		go -C $$dir build ./... || exit 1; \
	done

## clean: Clean up build artifacts and caches
.PHONY: clean
clean:
	@for dir in $(MODULE_DIRS); do \
		echo "Cleaning $$dir..."; \
		go -C $$dir clean -cache -testcache || exit 1; \
	done

## check: Run all checks (fmt, vet, lint, test)
.PHONY: check
check: fmt vet lint test
