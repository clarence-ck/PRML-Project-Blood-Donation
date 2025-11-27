output "ecr_repository_url" {
  description = "ECR repository URL for the donor API image"
  value       = aws_ecr_repository.app.repository_url
}

output "lambda_function_name" {
  description = "Name of the deployed Lambda function"
  value       = aws_lambda_function.app.function_name
}

output "http_api_invoke_url" {
  description = "Invoke URL for the HTTP API (proxying to Lambda)"
  value       = aws_apigatewayv2_stage.default.invoke_url
}
