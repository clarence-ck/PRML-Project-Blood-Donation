terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "ap-southeast-1"
}

variable "service_name" {
  description = "Base name for all deployed resources"
  type        = string
  default     = "donor-api"
}

variable "docker_image_tag" {
  description = "Docker image tag to deploy from ECR"
  type        = string
  default     = "latest"
}
