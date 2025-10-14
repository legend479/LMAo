"""
Email Automation Tool
Comprehensive email automation with SMTP integration, template system, and delivery tracking
"""

from typing import Dict, Any, List, Optional
import time
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formataddr
import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import re

from .registry import (
    BaseTool,
    ToolResult,
    ExecutionContext,
    ToolCapabilities,
    ResourceRequirements,
    ToolCapability,
)
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmailProvider:
    """Email provider configuration"""

    name: str
    smtp_server: str
    smtp_port: int
    use_tls: bool = True
    use_ssl: bool = False
    auth_required: bool = True
    max_attachment_size_mb: float = 25.0
    rate_limit_per_hour: int = 100


@dataclass
class EmailTemplate:
    """Email template configuration"""

    name: str
    subject_template: str
    body_template: str
    template_type: str = "text"  # text, html
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmailAttachment:
    """Email attachment information"""

    filename: str
    file_path: str
    content_type: str = "application/octet-stream"
    size_mb: float = 0.0
    is_valid: bool = True
    security_scan_result: Optional[Dict[str, Any]] = None


@dataclass
class EmailDeliveryStatus:
    """Email delivery status tracking"""

    message_id: str
    recipient: str
    status: str  # pending, sent, delivered, failed, bounced
    timestamp: datetime
    error_message: Optional[str] = None
    retry_count: int = 0
    provider_response: Optional[str] = None


@dataclass
class EmailSendResult:
    """Result of email sending operation"""

    success: bool
    message_id: str
    recipients_sent: List[str]
    recipients_failed: List[str]
    delivery_statuses: List[EmailDeliveryStatus]
    processing_time: float
    provider_used: str
    error_message: Optional[str] = None


class EmailProviderManager:
    """Manages multiple email providers with failover"""

    def __init__(self):
        self.providers: Dict[str, EmailProvider] = {}
        self.credentials: Dict[str, Dict[str, str]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_providers()

    def _initialize_default_providers(self):
        """Initialize common email providers"""

        # Gmail
        self.providers["gmail"] = EmailProvider(
            name="gmail",
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            use_tls=True,
            rate_limit_per_hour=100,
        )

        # Outlook/Hotmail
        self.providers["outlook"] = EmailProvider(
            name="outlook",
            smtp_server="smtp-mail.outlook.com",
            smtp_port=587,
            use_tls=True,
            rate_limit_per_hour=300,
        )

        # Yahoo
        self.providers["yahoo"] = EmailProvider(
            name="yahoo",
            smtp_server="smtp.mail.yahoo.com",
            smtp_port=587,
            use_tls=True,
            rate_limit_per_hour=100,
        )

        # Generic SMTP
        self.providers["smtp"] = EmailProvider(
            name="smtp",
            smtp_server="localhost",
            smtp_port=587,
            use_tls=True,
            rate_limit_per_hour=1000,
        )

    def configure_provider(
        self,
        provider_name: str,
        username: str,
        password: str,
        custom_config: Optional[Dict[str, Any]] = None,
    ):
        """Configure email provider credentials"""

        self.credentials[provider_name] = {"username": username, "password": password}

        # Apply custom configuration if provided
        if custom_config and provider_name in self.providers:
            provider = self.providers[provider_name]
            for key, value in custom_config.items():
                if hasattr(provider, key):
                    setattr(provider, key, value)

        # Initialize rate limiting
        self.rate_limits[provider_name] = {
            "sent_count": 0,
            "reset_time": datetime.now() + timedelta(hours=1),
        }

    def get_available_provider(self) -> Optional[str]:
        """Get an available provider that hasn't hit rate limits"""

        current_time = datetime.now()

        for provider_name, provider in self.providers.items():
            if provider_name not in self.credentials:
                continue

            # Check rate limits
            rate_limit_info = self.rate_limits.get(provider_name, {})

            # Reset rate limit if time has passed
            if current_time > rate_limit_info.get("reset_time", current_time):
                self.rate_limits[provider_name] = {
                    "sent_count": 0,
                    "reset_time": current_time + timedelta(hours=1),
                }
                rate_limit_info = self.rate_limits[provider_name]

            # Check if under rate limit
            if rate_limit_info.get("sent_count", 0) < provider.rate_limit_per_hour:
                return provider_name

        return None

    def increment_usage(self, provider_name: str):
        """Increment usage count for rate limiting"""
        if provider_name in self.rate_limits:
            self.rate_limits[provider_name]["sent_count"] += 1


class EmailTemplateManager:
    """Manages email templates with variable substitution"""

    def __init__(self):
        self.templates: Dict[str, EmailTemplate] = {}
        self._initialize_default_templates()

    def _initialize_default_templates(self):
        """Initialize default email templates"""

        # Document delivery template
        self.templates["document_delivery"] = EmailTemplate(
            name="document_delivery",
            subject_template="Document Generated: {document_title}",
            body_template="""
Hello {recipient_name},

I've generated the requested document for you.

Document Details:
- Title: {document_title}
- Format: {document_format}
- Generated: {generation_date}
- File Size: {file_size}

The document is attached to this email.

Best regards,
SE SME Agent
            """.strip(),
            template_type="text",
            variables=[
                "recipient_name",
                "document_title",
                "document_format",
                "generation_date",
                "file_size",
            ],
        )

        # Code analysis results template
        self.templates["code_analysis"] = EmailTemplate(
            name="code_analysis",
            subject_template="Code Analysis Results: {analysis_type}",
            body_template="""
Hello {recipient_name},

Your code analysis has been completed.

Analysis Summary:
- Type: {analysis_type}
- Files Analyzed: {file_count}
- Issues Found: {issue_count}
- Overall Score: {quality_score}

{analysis_summary}

Please find the detailed report attached.

Best regards,
SE SME Agent
            """.strip(),
            template_type="text",
            variables=[
                "recipient_name",
                "analysis_type",
                "file_count",
                "issue_count",
                "quality_score",
                "analysis_summary",
            ],
        )

        # Generic notification template
        self.templates["notification"] = EmailTemplate(
            name="notification",
            subject_template="{subject}",
            body_template="""
Hello {recipient_name},

{message_body}

{additional_info}

Best regards,
SE SME Agent
            """.strip(),
            template_type="text",
            variables=["recipient_name", "subject", "message_body", "additional_info"],
        )

    def get_template(self, template_name: str) -> EmailTemplate:
        """Get email template by name"""
        if template_name in self.templates:
            return self.templates[template_name]

        # Return generic template as fallback
        return self.templates.get("notification", self.templates["document_delivery"])

    def render_template(
        self, template_name: str, variables: Dict[str, Any]
    ) -> Dict[str, str]:
        """Render email template with variables"""

        template = self.get_template(template_name)

        # Substitute variables in subject
        subject = template.subject_template
        for var, value in variables.items():
            subject = subject.replace(f"{{{var}}}", str(value))

        # Substitute variables in body
        body = template.body_template
        for var, value in variables.items():
            body = body.replace(f"{{{var}}}", str(value))

        return {
            "subject": subject,
            "body": body,
            "template_type": template.template_type,
        }

    def register_template(self, template: EmailTemplate):
        """Register a new email template"""
        self.templates[template.name] = template


class AttachmentProcessor:
    """Processes and validates email attachments"""

    def __init__(self, max_total_size_mb: float = 25.0):
        self.max_total_size_mb = max_total_size_mb
        self.allowed_extensions = {
            ".pdf",
            ".docx",
            ".pptx",
            ".txt",
            ".md",
            ".json",
            ".csv",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".zip",
            ".tar.gz",
        }
        self.dangerous_extensions = {
            ".exe",
            ".bat",
            ".cmd",
            ".scr",
            ".pif",
            ".com",
            ".vbs",
            ".js",
        }

    def process_attachments(self, attachment_paths: List[str]) -> List[EmailAttachment]:
        """Process and validate attachments"""

        attachments = []
        total_size = 0.0

        for file_path in attachment_paths:
            attachment = self._process_single_attachment(file_path)

            if attachment.is_valid:
                total_size += attachment.size_mb

                # Check total size limit
                if total_size > self.max_total_size_mb:
                    attachment.is_valid = False
                    attachment.security_scan_result = {
                        "error": f"Total attachment size exceeds {self.max_total_size_mb}MB limit"
                    }

            attachments.append(attachment)

        return attachments

    def _process_single_attachment(self, file_path: str) -> EmailAttachment:
        """Process a single attachment"""

        filename = os.path.basename(file_path)
        file_extension = Path(file_path).suffix.lower()

        attachment = EmailAttachment(
            filename=filename,
            file_path=file_path,
            content_type=self._get_content_type(file_extension),
        )

        # Check if file exists
        if not os.path.exists(file_path):
            attachment.is_valid = False
            attachment.security_scan_result = {"error": "File does not exist"}
            return attachment

        # Get file size
        try:
            file_size_bytes = os.path.getsize(file_path)
            attachment.size_mb = file_size_bytes / (1024 * 1024)
        except OSError as e:
            attachment.is_valid = False
            attachment.security_scan_result = {"error": f"Cannot access file: {e}"}
            return attachment

        # Security checks
        security_result = self._perform_security_scan(file_path, file_extension)
        attachment.security_scan_result = security_result

        if not security_result.get("safe", True):
            attachment.is_valid = False

        return attachment

    def _get_content_type(self, file_extension: str) -> str:
        """Get MIME content type for file extension"""

        content_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".json": "application/json",
            ".csv": "text/csv",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".zip": "application/zip",
        }

        return content_types.get(file_extension, "application/octet-stream")

    def _perform_security_scan(
        self, file_path: str, file_extension: str
    ) -> Dict[str, Any]:
        """Perform basic security scanning on attachment"""

        scan_result = {
            "safe": True,
            "warnings": [],
            "scan_time": datetime.now().isoformat(),
        }

        # Check dangerous extensions
        if file_extension in self.dangerous_extensions:
            scan_result["safe"] = False
            scan_result["warnings"].append(
                f"Dangerous file extension: {file_extension}"
            )
            return scan_result

        # Check allowed extensions
        if file_extension not in self.allowed_extensions:
            scan_result["warnings"].append(f"Uncommon file extension: {file_extension}")

        # Check file size
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 50:  # Individual file size limit
                scan_result["safe"] = False
                scan_result["warnings"].append(f"File too large: {file_size_mb:.2f}MB")
        except OSError:
            scan_result["safe"] = False
            scan_result["warnings"].append("Cannot access file for size check")

        # Basic content scanning (simplified)
        try:
            with open(file_path, "rb") as f:
                # Read first 1KB for basic analysis
                header = f.read(1024)

                # Check for suspicious patterns (very basic)
                suspicious_patterns = [b"<script", b"javascript:", b"vbscript:"]
                for pattern in suspicious_patterns:
                    if pattern in header.lower():
                        scan_result["warnings"].append(
                            "Suspicious content pattern detected"
                        )
                        break

        except Exception as e:
            scan_result["warnings"].append(f"Content scan failed: {e}")

        return scan_result


class DeliveryTracker:
    """Tracks email delivery status and handles bounces"""

    def __init__(self):
        self.delivery_statuses: Dict[str, EmailDeliveryStatus] = {}
        self.bounce_patterns = [
            r"user unknown",
            r"mailbox unavailable",
            r"address rejected",
            r"domain not found",
            r"message too large",
        ]

    def create_delivery_status(
        self, message_id: str, recipient: str
    ) -> EmailDeliveryStatus:
        """Create initial delivery status"""

        status = EmailDeliveryStatus(
            message_id=message_id,
            recipient=recipient,
            status="pending",
            timestamp=datetime.now(),
        )

        self.delivery_statuses[f"{message_id}_{recipient}"] = status
        return status

    def update_status(
        self,
        message_id: str,
        recipient: str,
        new_status: str,
        error_message: Optional[str] = None,
        provider_response: Optional[str] = None,
    ):
        """Update delivery status"""

        key = f"{message_id}_{recipient}"
        if key in self.delivery_statuses:
            status = self.delivery_statuses[key]
            status.status = new_status
            status.timestamp = datetime.now()
            status.error_message = error_message
            status.provider_response = provider_response

            # Increment retry count for failed deliveries
            if new_status == "failed":
                status.retry_count += 1

    def check_bounce(self, error_message: str) -> bool:
        """Check if error message indicates a bounce"""

        if not error_message:
            return False

        error_lower = error_message.lower()
        return any(re.search(pattern, error_lower) for pattern in self.bounce_patterns)

    def get_delivery_summary(self) -> Dict[str, Any]:
        """Get summary of delivery statuses"""

        summary = {
            "total_messages": len(self.delivery_statuses),
            "status_counts": {},
            "bounce_count": 0,
            "retry_count": 0,
        }

        for status in self.delivery_statuses.values():
            # Count by status
            summary["status_counts"][status.status] = (
                summary["status_counts"].get(status.status, 0) + 1
            )

            # Count bounces
            if status.status == "bounced":
                summary["bounce_count"] += 1

            # Count retries
            summary["retry_count"] += status.retry_count

        return summary


class EmailAutomationTool(BaseTool):
    """Comprehensive email automation tool with SMTP integration and tracking"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.provider_manager = EmailProviderManager()
        self.template_manager = EmailTemplateManager()
        self.attachment_processor = AttachmentProcessor()
        self.delivery_tracker = DeliveryTracker()

        # Configuration
        self.max_recipients_per_email = (
            config.get("max_recipients_per_email", 50) if config else 50
        )
        self.retry_attempts = config.get("retry_attempts", 3) if config else 3
        self.retry_delay_seconds = (
            config.get("retry_delay_seconds", 60) if config else 60
        )

    async def initialize(self):
        """Initialize the email automation tool"""
        await super().initialize()

        # Configure default provider if credentials are available in environment
        self._configure_from_environment()

        logger.info(
            "Email Automation Tool initialized",
            providers_configured=len(self.provider_manager.credentials),
            templates_available=len(self.template_manager.templates),
        )

    def _configure_from_environment(self):
        """Configure email providers from environment variables"""

        # Check for Gmail configuration
        gmail_user = os.getenv("GMAIL_USERNAME")
        gmail_pass = os.getenv("GMAIL_PASSWORD") or os.getenv("GMAIL_APP_PASSWORD")

        if gmail_user and gmail_pass:
            self.provider_manager.configure_provider("gmail", gmail_user, gmail_pass)
            logger.info("Gmail provider configured from environment")

        # Check for SMTP configuration
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_user = os.getenv("SMTP_USERNAME")
        smtp_pass = os.getenv("SMTP_PASSWORD")
        smtp_port = os.getenv("SMTP_PORT", "587")

        if smtp_server and smtp_user and smtp_pass:
            custom_config = {"smtp_server": smtp_server, "smtp_port": int(smtp_port)}
            self.provider_manager.configure_provider(
                "smtp", smtp_user, smtp_pass, custom_config
            )
            logger.info("SMTP provider configured from environment")

    async def execute(
        self, parameters: Dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        """Execute email sending with tracking and error handling"""

        start_time = time.time()

        try:
            # Extract parameters
            recipients = parameters["recipients"]
            subject = parameters.get("subject", "")
            body = parameters.get("body", "")
            template_name = parameters.get("template")
            template_variables = parameters.get("template_variables", {})
            attachments = parameters.get("attachments", [])
            sender_name = parameters.get("sender_name", "SE SME Agent")
            priority = parameters.get("priority", "normal")

            logger.info(
                "Executing email automation",
                recipients_count=len(recipients),
                has_attachments=len(attachments) > 0,
                template=template_name,
                session_id=context.session_id,
            )

            # Validate recipients
            valid_recipients = self._validate_recipients(recipients)
            if not valid_recipients:
                raise ValueError("No valid recipients provided")

            # Process template if specified
            if template_name:
                rendered = self.template_manager.render_template(
                    template_name, template_variables
                )
                subject = rendered["subject"]
                body = rendered["body"]

            # Process attachments
            processed_attachments = []
            if attachments:
                processed_attachments = self.attachment_processor.process_attachments(
                    attachments
                )

                # Check for invalid attachments
                invalid_attachments = [
                    att for att in processed_attachments if not att.is_valid
                ]
                if invalid_attachments:
                    logger.warning(
                        "Some attachments are invalid",
                        invalid_count=len(invalid_attachments),
                    )

            # Send emails
            send_result = await self._send_emails(
                recipients=valid_recipients,
                subject=subject,
                body=body,
                attachments=processed_attachments,
                sender_name=sender_name,
                priority=priority,
                context=context,
            )

            execution_time = time.time() - start_time

            # Prepare result data
            result_data = {
                "message_id": send_result.message_id,
                "recipients_sent": send_result.recipients_sent,
                "recipients_failed": send_result.recipients_failed,
                "total_recipients": len(recipients),
                "success_count": len(send_result.recipients_sent),
                "failure_count": len(send_result.recipients_failed),
                "attachments_processed": len(processed_attachments),
                "valid_attachments": len(
                    [att for att in processed_attachments if att.is_valid]
                ),
                "provider_used": send_result.provider_used,
                "delivery_tracking": {
                    "tracking_enabled": True,
                    "delivery_statuses": [
                        {
                            "recipient": status.recipient,
                            "status": status.status,
                            "timestamp": status.timestamp.isoformat(),
                            "error_message": status.error_message,
                        }
                        for status in send_result.delivery_statuses
                    ],
                },
                "processing_time": execution_time,
            }

            # Add attachment details if present
            if processed_attachments:
                result_data["attachment_details"] = [
                    {
                        "filename": att.filename,
                        "size_mb": att.size_mb,
                        "is_valid": att.is_valid,
                        "security_warnings": (
                            att.security_scan_result.get("warnings", [])
                            if att.security_scan_result
                            else []
                        ),
                    }
                    for att in processed_attachments
                ]

            # Calculate quality and confidence scores
            quality_score = self._calculate_quality_score(
                send_result, processed_attachments
            )
            confidence_score = self._calculate_confidence_score(
                send_result, valid_recipients
            )

            result = ToolResult(
                data=result_data,
                metadata={
                    "tool": "email_automation",
                    "version": "1.0.0",
                    "provider_used": send_result.provider_used,
                    "template_used": template_name,
                    "session_id": context.session_id,
                    "priority": priority,
                },
                execution_time=execution_time,
                success=send_result.success,
                resource_usage={
                    "cpu_usage": 0.2,
                    "memory_usage_mb": 32,
                    "network_requests": len(valid_recipients),
                },
                quality_score=quality_score,
                confidence_score=confidence_score,
                error_message=send_result.error_message,
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                "Email automation failed", error=str(e), session_id=context.session_id
            )

            return ToolResult(
                data=None,
                metadata={
                    "tool": "email_automation",
                    "session_id": context.session_id,
                    "error_type": type(e).__name__,
                },
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                quality_score=0.0,
                confidence_score=0.0,
            )

    def _validate_recipients(self, recipients: List[str]) -> List[str]:
        """Validate email recipients"""

        valid_recipients = []
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

        for recipient in recipients:
            recipient = recipient.strip()
            if email_pattern.match(recipient):
                valid_recipients.append(recipient)
            else:
                logger.warning(f"Invalid email address: {recipient}")

        return valid_recipients

    async def _send_emails(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        attachments: List[EmailAttachment],
        sender_name: str,
        priority: str,
        context: ExecutionContext,
    ) -> EmailSendResult:
        """Send emails with retry logic and tracking"""

        # Get available provider
        provider_name = self.provider_manager.get_available_provider()
        if not provider_name:
            raise RuntimeError(
                "No email provider available or all providers have hit rate limits"
            )

        provider = self.provider_manager.providers[provider_name]
        credentials = self.provider_manager.credentials[provider_name]

        # Generate message ID
        message_id = self._generate_message_id()

        # Initialize tracking
        delivery_statuses = []
        for recipient in recipients:
            status = self.delivery_tracker.create_delivery_status(message_id, recipient)
            delivery_statuses.append(status)

        recipients_sent = []
        recipients_failed = []

        try:
            # Create SMTP connection
            smtp_server = await self._create_smtp_connection(provider, credentials)

            # Send to each recipient (or batch if provider supports it)
            for recipient in recipients:
                try:
                    # Create message
                    message = self._create_message(
                        sender_email=credentials["username"],
                        sender_name=sender_name,
                        recipient=recipient,
                        subject=subject,
                        body=body,
                        attachments=attachments,
                        message_id=message_id,
                    )

                    # Send message
                    smtp_server.send_message(message)

                    # Update tracking
                    self.delivery_tracker.update_status(message_id, recipient, "sent")
                    recipients_sent.append(recipient)

                    logger.debug(f"Email sent successfully to {recipient}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to send email to {recipient}: {error_msg}")

                    # Check if it's a bounce
                    if self.delivery_tracker.check_bounce(error_msg):
                        self.delivery_tracker.update_status(
                            message_id, recipient, "bounced", error_msg
                        )
                    else:
                        self.delivery_tracker.update_status(
                            message_id, recipient, "failed", error_msg
                        )

                    recipients_failed.append(recipient)

            # Close SMTP connection
            smtp_server.quit()

            # Update provider usage
            self.provider_manager.increment_usage(provider_name)

            # Determine overall success
            success = len(recipients_sent) > 0

            return EmailSendResult(
                success=success,
                message_id=message_id,
                recipients_sent=recipients_sent,
                recipients_failed=recipients_failed,
                delivery_statuses=delivery_statuses,
                processing_time=0.0,  # Will be set by caller
                provider_used=provider_name,
                error_message=None if success else "Failed to send to all recipients",
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"SMTP connection or sending failed: {error_msg}")

            # Mark all recipients as failed
            for recipient in recipients:
                self.delivery_tracker.update_status(
                    message_id, recipient, "failed", error_msg
                )
                recipients_failed.append(recipient)

            return EmailSendResult(
                success=False,
                message_id=message_id,
                recipients_sent=[],
                recipients_failed=recipients_failed,
                delivery_statuses=delivery_statuses,
                processing_time=0.0,
                provider_used=provider_name,
                error_message=error_msg,
            )

    async def _create_smtp_connection(
        self, provider: EmailProvider, credentials: Dict[str, str]
    ) -> smtplib.SMTP:
        """Create SMTP connection with proper authentication"""

        try:
            # Create SMTP connection
            if provider.use_ssl:
                smtp_server = smtplib.SMTP_SSL(provider.smtp_server, provider.smtp_port)
            else:
                smtp_server = smtplib.SMTP(provider.smtp_server, provider.smtp_port)

            # Enable debug if needed
            # smtp_server.set_debuglevel(1)

            # Start TLS if required
            if provider.use_tls and not provider.use_ssl:
                smtp_server.starttls(context=ssl.create_default_context())

            # Authenticate if required
            if provider.auth_required:
                smtp_server.login(credentials["username"], credentials["password"])

            return smtp_server

        except Exception as e:
            logger.error(f"Failed to create SMTP connection: {e}")
            raise

    def _create_message(
        self,
        sender_email: str,
        sender_name: str,
        recipient: str,
        subject: str,
        body: str,
        attachments: List[EmailAttachment],
        message_id: str,
    ) -> MIMEMultipart:
        """Create email message with attachments"""

        # Create message
        message = MIMEMultipart()

        # Set headers
        message["From"] = formataddr((sender_name, sender_email))
        message["To"] = recipient
        message["Subject"] = subject
        message["Message-ID"] = f"<{message_id}@se-sme-agent>"
        message["X-Mailer"] = "SE SME Agent Email Tool v1.0"

        # Add body
        message.attach(MIMEText(body, "plain"))

        # Add attachments
        for attachment in attachments:
            if attachment.is_valid:
                try:
                    with open(attachment.file_path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())

                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {attachment.filename}",
                    )

                    message.attach(part)

                except Exception as e:
                    logger.warning(f"Failed to attach file {attachment.filename}: {e}")

        return message

    def _generate_message_id(self) -> str:
        """Generate unique message ID"""

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_part = hashlib.md5(f"{timestamp}{time.time()}".encode()).hexdigest()[:8]
        return f"{timestamp}_{random_part}"

    def _calculate_quality_score(
        self, send_result: EmailSendResult, attachments: List[EmailAttachment]
    ) -> float:
        """Calculate quality score for email sending"""

        base_score = 0.7

        # Success rate factor
        total_recipients = len(send_result.recipients_sent) + len(
            send_result.recipients_failed
        )
        if total_recipients > 0:
            success_rate = len(send_result.recipients_sent) / total_recipients
            base_score += success_rate * 0.2

        # Attachment processing factor
        if attachments:
            valid_attachments = len([att for att in attachments if att.is_valid])
            attachment_success_rate = valid_attachments / len(attachments)
            base_score += attachment_success_rate * 0.1

        return min(1.0, base_score)

    def _calculate_confidence_score(
        self, send_result: EmailSendResult, recipients: List[str]
    ) -> float:
        """Calculate confidence score for email sending"""

        base_confidence = 0.8

        # Provider availability factor
        if send_result.provider_used:
            base_confidence += 0.1

        # Recipient validation factor
        if len(recipients) > 0:
            base_confidence += 0.1

        # Reduce confidence for failures
        if send_result.recipients_failed:
            failure_rate = len(send_result.recipients_failed) / len(recipients)
            base_confidence -= failure_rate * 0.3

        return max(0.0, min(1.0, base_confidence))

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for email automation"""
        return {
            "name": "email_automation",
            "description": "Send emails with SMTP integration, template support, attachment processing, and delivery tracking",
            "version": "1.0.0",
            "parameters": {
                "recipients": {
                    "type": "array",
                    "items": {"type": "string", "format": "email"},
                    "description": "List of email recipients",
                    "required": True,
                    "minItems": 1,
                    "maxItems": 50,
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line",
                    "required": False,
                    "maxLength": 200,
                },
                "body": {
                    "type": "string",
                    "description": "Email body content",
                    "required": False,
                    "maxLength": 10000,
                },
                "template": {
                    "type": "string",
                    "description": "Email template to use",
                    "enum": ["document_delivery", "code_analysis", "notification"],
                    "required": False,
                },
                "template_variables": {
                    "type": "object",
                    "description": "Variables for template substitution",
                    "required": False,
                },
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to attach",
                    "required": False,
                    "maxItems": 10,
                },
                "sender_name": {
                    "type": "string",
                    "description": "Display name for sender",
                    "default": "SE SME Agent",
                    "required": False,
                    "maxLength": 100,
                },
                "priority": {
                    "type": "string",
                    "description": "Email priority level",
                    "enum": ["low", "normal", "high"],
                    "default": "normal",
                    "required": False,
                },
            },
            "required_params": ["recipients"],
            "returns": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "recipients_sent": {"type": "array", "items": {"type": "string"}},
                    "recipients_failed": {"type": "array", "items": {"type": "string"}},
                    "total_recipients": {"type": "integer"},
                    "success_count": {"type": "integer"},
                    "failure_count": {"type": "integer"},
                    "attachments_processed": {"type": "integer"},
                    "valid_attachments": {"type": "integer"},
                    "provider_used": {"type": "string"},
                    "delivery_tracking": {
                        "type": "object",
                        "properties": {
                            "tracking_enabled": {"type": "boolean"},
                            "delivery_statuses": {"type": "array"},
                        },
                    },
                    "attachment_details": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
            "capabilities": {
                "primary": "communication",
                "secondary": ["document_delivery", "notification"],
                "input_types": ["text", "email_addresses", "file_paths"],
                "output_types": ["delivery_status", "tracking_info"],
                "supported_providers": ["gmail", "outlook", "yahoo", "smtp"],
            },
        }

    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            primary_capability=ToolCapability.COMMUNICATION,
            secondary_capabilities=[
                ToolCapability.DOCUMENT_GENERATION,
                ToolCapability.VALIDATION,
            ],
            input_types=["text", "email", "file_path", "template"],
            output_types=["delivery_status", "tracking_report", "json"],
            supported_formats=["text", "html", "attachments"],
            language_support=["en"],
        )

    def get_resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements"""
        return ResourceRequirements(
            cpu_cores=0.5,
            memory_mb=128,
            network_bandwidth_mbps=5.0,  # For SMTP connections
            storage_mb=10,  # For temporary attachment processing
            gpu_memory_mb=0,
            max_execution_time=120,  # Allow time for multiple recipients
            concurrent_limit=5,  # Limit concurrent email operations
        )

    # Additional utility methods

    def configure_provider(
        self,
        provider_name: str,
        username: str,
        password: str,
        custom_config: Optional[Dict[str, Any]] = None,
    ):
        """Configure email provider (for external configuration)"""
        self.provider_manager.configure_provider(
            provider_name, username, password, custom_config
        )

    def add_template(self, template: EmailTemplate):
        """Add custom email template"""
        self.template_manager.register_template(template)

    def get_delivery_summary(self) -> Dict[str, Any]:
        """Get delivery tracking summary"""
        return self.delivery_tracker.get_delivery_summary()

    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of configured providers"""
        status = {}

        for provider_name in self.provider_manager.providers:
            is_configured = provider_name in self.provider_manager.credentials
            rate_limit_info = self.provider_manager.rate_limits.get(provider_name, {})

            status[provider_name] = {
                "configured": is_configured,
                "available": is_configured
                and self.provider_manager.get_available_provider() == provider_name,
                "sent_count": rate_limit_info.get("sent_count", 0),
                "rate_limit": self.provider_manager.providers[
                    provider_name
                ].rate_limit_per_hour,
                "reset_time": (
                    rate_limit_info.get("reset_time").isoformat()
                    if rate_limit_info.get("reset_time")
                    else None
                ),
            }

        return status

    async def cleanup(self):
        """Cleanup tool resources"""
        logger.info("Cleaning up Email Automation Tool")

        # Clear delivery tracking history older than 24 hours
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            expired_keys = []

            for key, status in self.delivery_tracker.delivery_statuses.items():
                if status.timestamp < cutoff_time:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.delivery_tracker.delivery_statuses[key]

            if expired_keys:
                logger.debug(
                    f"Cleaned up {len(expired_keys)} expired delivery statuses"
                )

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
