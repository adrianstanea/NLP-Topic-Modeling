.PHONY: all report presentation clean help

BASE_DIR := $(shell pwd)

REPORT_SRC_DIR := report
REPORT_BUILD_DIR := $(BASE_DIR)/build_report

PRESENTATION_SRC_DIR := presentation
PRESENTATION_BUILD_DIR := $(BASE_DIR)/build_presentation

help: ## Show this help menu
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

report: ## Build the PDF report
	mkdir -p $(REPORT_BUILD_DIR)
	cd $(REPORT_SRC_DIR) && \
	pdflatex -output-directory=$(REPORT_BUILD_DIR) report.tex && \
	pdflatex -output-directory=$(REPORT_BUILD_DIR) report.tex
	cp $(REPORT_BUILD_DIR)/report.pdf $(BASE_DIR)/report.pdf

presentation: ## Build the presentation slides
	mkdir -p $(PRESENTATION_BUILD_DIR)
	cd $(PRESENTATION_SRC_DIR) && \
	pdflatex -output-directory=$(PRESENTATION_BUILD_DIR) presentation.tex && \
	pdflatex -output-directory=$(PRESENTATION_BUILD_DIR) presentation.tex
	cp $(PRESENTATION_BUILD_DIR)/presentation.pdf $(BASE_DIR)/presentation.pdf

clean: ## Remove build artifacts
	rm -rf $(REPORT_BUILD_DIR) $(PRESENTATION_BUILD_DIR)


all: clean report presentation ## Clean and rebuild all documents