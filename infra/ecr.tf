resource "aws_ecr_repository" "app" {
  name = var.service_name

  image_scanning_configuration {
    scan_on_push = true
  }

  force_delete = true
}
