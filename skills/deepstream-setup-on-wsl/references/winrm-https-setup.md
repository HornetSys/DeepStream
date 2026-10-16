# WinRM HTTPS prerequisite for a public remote target

Use WinRM HTTPS only when the Windows machine already has a server-authentication
certificate whose subject/SAN matches the hostname used by the controller. The
controller must trust the issuing CA. Do not disable certificate validation.

Run the following in an elevated PowerShell terminal on the Windows target after
review and explicit user confirmation. Replace the placeholders:

```powershell
$thumbprint = '<SERVER_CERTIFICATE_THUMBPRINT>'
$controller = '<CONTROLLER_IP_OR_SUBNET>'
$certificate = Get-Item "Cert:\LocalMachine\My\$thumbprint"
if (-not $certificate.HasPrivateKey) { throw 'certificate has no private key' }
if ($certificate.NotAfter -le (Get-Date)) { throw 'certificate is expired' }

$existing = Get-ChildItem WSMan:\localhost\Listener |
  Where-Object { $_.Keys -contains 'Transport=HTTPS' }
if ($existing) {
  throw 'an HTTPS listener already exists; inspect it instead of replacing it automatically'
}

New-Item -Path WSMan:\localhost\Listener -Transport HTTPS -Address * `
  -CertificateThumbPrint $thumbprint -Force
New-NetFirewallRule -DisplayName 'DeepStream WSL WinRM HTTPS' `
  -Direction Inbound -Action Allow -Protocol TCP -LocalPort 5986 `
  -RemoteAddress $controller

Get-ChildItem WSMan:\localhost\Listener
Get-NetFirewallRule -DisplayName 'DeepStream WSL WinRM HTTPS' |
  Get-NetFirewallAddressFilter
```

From the controller, connect using the certificate hostname rather than a raw IP
unless the IP is present in the certificate SAN. If secure WinRM cannot be
configured, use local guide mode. Do not fall back to external HTTP port 5985.
